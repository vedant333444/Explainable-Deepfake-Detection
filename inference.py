from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN
from transformers import AutoImageProcessor

from model_def import load_model, MODEL_NAME

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "dinov2_phase2_best_2500.pt"

NUM_FRAMES = 15
THRESHOLD = 0.50
FACE_MARGIN = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model, device = load_model(str(MODEL_PATH), device=device, model_name=MODEL_NAME)

mtcnn = MTCNN(
    keep_all=False,
    device=device,
    min_face_size=40
)


def sample_frames(video_path: str, num_frames: int = NUM_FRAMES):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames > 0:
        indices = np.linspace(0, total_frames - 1, num=min(num_frames, total_frames), dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((int(idx), frame))
    else:
        idx = 0
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % 5 == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((idx, frame))
            idx += 1

    cap.release()
    return frames[:num_frames]


def crop_face(rgb_frame: np.ndarray, margin: int = FACE_MARGIN):
    pil_img = Image.fromarray(rgb_frame)

    boxes, probs = mtcnn.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return pil_img.resize((224, 224)), False

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best_idx = int(np.argmax(areas))
    box = boxes[best_idx].astype(int)

    x1, y1, x2, y2 = box.tolist()
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(pil_img.width, x2 + margin)
    y2 = min(pil_img.height, y2 + margin)

    if x2 <= x1 or y2 <= y1:
        return pil_img.resize((224, 224)), False

    crop = pil_img.crop((x1, y1, x2, y2)).resize((224, 224))
    return crop, True


@torch.no_grad()
def predict_video(video_path: str, num_frames: int = NUM_FRAMES, threshold: float = THRESHOLD, top_k: int = 3):
    frames = sample_frames(video_path, num_frames=num_frames)
    if not frames:
        raise ValueError("No frames could be sampled from the uploaded video.")

    face_images = []
    frame_indices = []

    for idx, frame in frames:
        face_img, _ = crop_face(frame)
        face_images.append(face_img)
        frame_indices.append(idx)

    inputs = processor(images=face_images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    logits = model(pixel_values)
    probs = torch.sigmoid(logits).detach().cpu().numpy()

    video_prob = float(np.mean(probs))
    label = "Fake" if video_prob >= threshold else "Real"

    order = np.argsort(-probs)
    top_images = []
    lines = []

    for rank in order[:top_k]:
        top_images.append(face_images[rank])
        lines.append(f"Frame {frame_indices[rank]}: fake_prob={probs[rank]:.4f}")

    return {
        "label": label,
        "confidence": video_prob,
        "num_frames": len(frames),
        "threshold": threshold,
        "top_images": top_images,
        "frame_scores_text": "\n".join(lines),
        "frame_probabilities": probs.tolist(),
        "frame_indices": frame_indices,
    }