import time
from pathlib import Path

import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from inference import predict_video

BASE_DIR = Path(__file__).resolve().parent
TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)


def build_frame_plot(frame_indices, frame_probs):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(frame_indices, frame_probs, marker="o", linewidth=2)
    ax.set_title("Frame-wise Fake Probability")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Fake Probability")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def build_report_html(label, confidence, num_frames, threshold, frame_text):
    status_color = "#d9534f" if label.lower() == "fake" else "#5cb85c"
    title = "FAKE VIDEO DETECTED" if label.lower() == "fake" else "REAL VIDEO"

    html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                background: #111827;
                color: #f3f4f6;
                padding: 24px;
            }}
            .card {{
                background: #1f2937;
                border: 1px solid #374151;
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .banner {{
                background: {status_color};
                color: white;
                padding: 16px 20px;
                border-radius: 14px;
                margin-bottom: 18px;
            }}
            h1, h2 {{
                margin: 0 0 10px 0;
            }}
            .meta {{
                line-height: 1.8;
                font-size: 15px;
            }}
            pre {{
                background: #0f172a;
                padding: 14px;
                border-radius: 12px;
                overflow-x: auto;
                border: 1px solid #334155;
            }}
            .small {{
                color: #cbd5e1;
                font-size: 13px;
            }}
        </style>
    </head>
    <body>
        <div class="banner">
            <h1>{title}</h1>
            <div>The uploaded video is analyzed using the fine-tuned DINOv2 model.</div>
        </div>

        <div class="card">
            <h2>Prediction Summary</h2>
            <div class="meta">
                <b>Prediction:</b> {label}<br>
                <b>Fake Probability:</b> {confidence:.4f}<br>
                <b>Frames Used:</b> {num_frames}<br>
                <b>Threshold:</b> {threshold:.2f}
            </div>
        </div>

        <div class="card">
            <h2>Top Suspicious Frames</h2>
            <pre>{frame_text}</pre>
            <div class="small">The app highlights the frames with the highest fake probabilities.</div>
        </div>
    </body>
    </html>
    """

    report_path = TEMP_DIR / "deepfake_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(report_path)


def analyze_video(video_input, progress=gr.Progress()):
    start_time = time.time()

    if video_input is None:
        empty_df = pd.DataFrame(columns=["Frame", "Fake Probability"])
        return (
            "<div class='result-card neutral'><h2>No video uploaded</h2><p>Please upload a video first.</p></div>",
            0.0,
            [],
            empty_df,
            "",
            None,
            "Status: Waiting for upload...",
            0.0,
            None,
        )

    progress(0.1, desc="Preparing video")

    if isinstance(video_input, dict):
        video_path = video_input.get("name") or video_input.get("path")
    elif isinstance(video_input, tuple):
        video_path = video_input[0]
    else:
        video_path = video_input

    progress(0.35, desc="Running deepfake detection")
    result = predict_video(video_path)
    progress(0.85, desc="Preparing outputs")

    label = result["label"]
    confidence = float(result["confidence"])
    confidence_percent = confidence * 100.0
    frame_indices = result.get("frame_indices", list(range(len(result["frame_probabilities"]))))
    frame_probs = result["frame_probabilities"]

    if label.lower() == "fake":
        badge_class = "fake"
        badge_title = "FAKE VIDEO DETECTED"
        badge_subtitle = "The uploaded video is highly likely to be manipulated."
    else:
        badge_class = "real"
        badge_title = "REAL VIDEO"
        badge_subtitle = "The uploaded video appears authentic based on the current model."

    verdict_html = f"""
    <div class="result-card {badge_class}">
        <h2>{badge_title}</h2>
        <p>{badge_subtitle}</p>
        <div class="score-line"><strong>Fake Probability:</strong> {confidence:.4f}</div>
        <div class="score-line"><strong>Frames Used:</strong> {result["num_frames"]}</div>
        <div class="score-line"><strong>Threshold:</strong> {result["threshold"]:.2f}</div>
    </div>
    """

    frame_lines = result["frame_scores_text"].split("\n") if result["frame_scores_text"] else []
    frame_rows = []
    for line in frame_lines:
        if not line.strip():
            continue
        try:
            left, right = line.split(":")
            frame_num = left.replace("Frame", "").strip()
            prob = right.replace("fake_prob=", "").strip()
            frame_rows.append([frame_num, prob])
        except Exception:
            frame_rows.append([line, ""])

    frame_table = pd.DataFrame(frame_rows, columns=["Frame", "Fake Probability"])

    frame_plot = build_frame_plot(frame_indices, frame_probs)
    report_path = build_report_html(
        label=label,
        confidence=confidence,
        num_frames=result["num_frames"],
        threshold=result["threshold"],
        frame_text=result["frame_scores_text"],
    )

    elapsed = time.time() - start_time
    status_text = f"Status: Completed in {elapsed:.2f} seconds"
    progress(1.0, desc="Done")

    return (
        verdict_html,
        confidence,
        result["top_images"],
        frame_table,
        result["frame_scores_text"],
        str(report_path),
        status_text,
        confidence_percent,
        frame_plot,
    )


css = """
body {
    background: #0f1117;
}

#title {
    text-align: center;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 0;
}

#subtitle {
    text-align: center;
    font-size: 16px;
    margin-top: 0;
    margin-bottom: 20px;
    color: #cfcfcf;
}

.result-card {
    border-radius: 16px;
    padding: 18px;
    margin-bottom: 14px;
    border: 1px solid #333;
    color: white;
}

.result-card.fake {
    background: linear-gradient(135deg, #7a1f1f, #2b1111);
    border-color: #ff6b6b;
}

.result-card.real {
    background: linear-gradient(135deg, #1f6b3b, #10261b);
    border-color: #3ddc97;
}

.result-card.neutral {
    background: #222;
}

.result-card h2 {
    margin: 0 0 6px 0;
    font-size: 26px;
}

.result-card p {
    margin: 0 0 12px 0;
    opacity: 0.92;
}

.score-line {
    margin: 6px 0;
    font-size: 15px;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    margin-top: 8px;
}
"""


with gr.Blocks(css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        <div id="title">Explainable Deepfake Detection System</div>
        <div id="subtitle">Upload a video and get a real/fake prediction with confidence, suspicious frames, and a downloadable report.</div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            analyze_btn = gr.Button("Analyze Video", variant="primary")

            gr.Markdown(
                """
                <div class="section-title">How it works</div>
                <ol>
                    <li>Upload a video</li>
                    <li>The system samples frames</li>
                    <li>Faces are detected with MTCNN</li>
                    <li>DINOv2 predicts each frame</li>
                    <li>Final video-level result is shown</li>
                </ol>
                """
            )

        with gr.Column(scale=1):
            verdict_html = gr.HTML()
            confidence_out = gr.Number(label="Fake Probability")
            confidence_bar = gr.Slider(
                label="Confidence (%)",
                minimum=0,
                maximum=100,
                step=1,
                value=0,
                interactive=False,
            )
            status_out = gr.Markdown("Status: Ready")
            report_file = gr.File(label="Download Report")

    with gr.Row():
        gallery_out = gr.Gallery(label="Top Suspicious Frames", columns=3, height=320)

    with gr.Row():
        frame_table_out = gr.Dataframe(
            label="Suspicious Frame Table",
            headers=["Frame", "Fake Probability"],
            datatype=["str", "str"],
            row_count=3,
            col_count=(2, "fixed"),
            interactive=False,
        )

    with gr.Row():
        frame_plot_out = gr.Plot(label="Frame Probability Timeline")

    frame_text = gr.Textbox(label="Frame Scores", lines=5)

    analyze_btn.click(
        fn=analyze_video,
        inputs=video_input,
        outputs=[
            verdict_html,
            confidence_out,
            gallery_out,
            frame_table_out,
            frame_text,
            report_file,
            status_out,
            confidence_bar,
            frame_plot_out,
        ],
    )

    demo.queue()

if __name__ == "__main__":
    demo.launch()