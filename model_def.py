import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "facebook/dinov2-base"


class DinoForDeepfake(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_token).squeeze(1)
        return logits


def load_model(checkpoint_path: str, device=None, model_name: str = MODEL_NAME):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DinoForDeepfake(model_name=model_name).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    # Handle a few common checkpoint formats
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt:
                ckpt = ckpt[key]
                break

    model.load_state_dict(ckpt)
    model.eval()

    return model, device