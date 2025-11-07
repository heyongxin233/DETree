"""Gradio-based web demo for DETree kNN inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import gradio as gr

# Ensure imports resolve when running from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from detree.inference import Detector  # noqa: E402


def build_interface(detector: Detector) -> gr.Blocks:
    """Create a Gradio Blocks UI that wraps the detector."""

    def _predict(text: str):
        text = (text or "").strip()
        if not text:
            empty_scores = {
                "AI（阈值归一化概率）": 0.0,
                "Human（阈值归一化概率）": 0.0,
            }
            return "", empty_scores, "Please enter text to analyse."

        token_count = len(detector.tokenizer.encode(text, add_special_tokens=True))

        prediction = detector.predict([text])[0]
        probability_ai = float(prediction.probability_ai)
        probability_human = float(prediction.probability_human)
        label = prediction.label

        def _threshold_scaled_probability(
            human_probability: float, threshold: float
        ) -> float:
            """Project the raw probability onto a threshold-centred scale.

            The transformation keeps the human/AI probabilities summing to 1, while
            mapping the configured threshold to 0.5 so that the scoreboard's split
            visually matches the decision boundary.
            """

            eps = 1e-9
            # Avoid numerical issues when the threshold is extremely close to 0 or 1.
            clipped_threshold = min(max(threshold, eps), 1.0 - eps)
            if human_probability >= clipped_threshold:
                scaled = 0.5 + 0.5 * (
                    (human_probability - clipped_threshold) / (1.0 - clipped_threshold)
                )
            else:
                scaled = 0.5 * (human_probability / clipped_threshold)
            return float(min(max(scaled, 0.0), 1.0))

        scaled_human_probability = _threshold_scaled_probability(
            probability_human, detector.threshold
        )
        scaled_ai_probability = 1.0 - scaled_human_probability

        decision_scores = {
            "AI（阈值归一化概率）": scaled_ai_probability,
            "Human（阈值归一化概率）": scaled_human_probability,
        }

        explanation_lines = [
            f"Top-k neighbours: {detector.top_k}",
            f"Layer: {detector.layer}",
            f"Input tokens: {token_count}",
            "阈值：{threshold:.2f}（当人工概率≥该阈值时判定为人工文本）".format(
                threshold=detector.threshold
            ),
            "AI 概率：{ai:.2%}，人工概率：{human:.2%}".format(
                ai=probability_ai, human=probability_human
            ),
            "人工概率与阈值差值：{margin:+.2%}".format(
                margin=probability_human - detector.threshold
            ),
            "阈值归一化后概率（AI / Human）：{ai_scaled:.2%} / {human_scaled:.2%}".format(
                ai_scaled=scaled_ai_probability,
                human_scaled=scaled_human_probability,
            ),
        ]
        if token_count < 50:
            explanation_lines.append(
                f"警告：输入 token 数量（{token_count}）少于 50，结果可能不准确。"
            )
        explanation = "\n".join(explanation_lines)
        return label, decision_scores, explanation

    def _change_layer(layer_value: str):
        layer_value = (layer_value or "").strip()
        if not layer_value:
            return gr.update(value="Please select a valid layer.")
        try:
            numeric_layer = int(layer_value)
        except ValueError:
            return gr.update(value="Layer change failed: layer must be an integer.")
        try:
            detector.set_layer(numeric_layer)
        except ValueError as exc:
            return gr.update(value=f"Layer change failed: {exc}")
        return gr.update(value=f"Switched to layer {detector.layer}.")

    def _change_top_k(top_k_value):
        if top_k_value is None:
            message = "Top-k change failed: please enter an integer between 1 and 100."
            return gr.update(value=message), gr.update(value=detector.top_k)
        try:
            numeric_top_k = int(top_k_value)
        except (TypeError, ValueError):
            message = "Top-k change failed: value must be an integer."
            return gr.update(value=message), gr.update(value=detector.top_k)
        if not 1 <= numeric_top_k <= 100:
            message = "Top-k change failed: value must be between 1 and 100."
            return gr.update(value=message), gr.update(value=detector.top_k)
        detector.top_k = numeric_top_k
        return (
            gr.update(value=f"Top-k set to {detector.top_k}."),
            gr.update(value=detector.top_k),
        )

    with gr.Blocks(title="DETree AI Text Detector") as demo:
        gr.Markdown(
            "# DETree AI Text Detector\n"
            "当前判定阈值：{threshold:.2f}（人工概率≥阈值时判定为人工文本）".format(
                threshold=detector.threshold
            )
        )
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    lines=8,
                    label="Input text",
                    placeholder="Paste your sample here...",
                )
                layer_selector = gr.Dropdown(
                    choices=[str(layer) for layer in detector.get_available_layers()],
                    value=str(detector.layer),
                    label="Database layer",
                )
                top_k_input = gr.Number(
                    value=detector.top_k,
                    precision=0,
                    label="Top-k neighbours (1-100)",
                )
                submit_btn = gr.Button("Analyse")
            with gr.Column():
                label_output = gr.Textbox(label="Prediction", interactive=False)
                probs_output = gr.Label(label="阈值判定看板")
                meta_output = gr.Markdown()

        submit_btn.click(
            _predict,
            inputs=input_text,
            outputs=[label_output, probs_output, meta_output],
        )

        layer_selector.change(
            _change_layer,
            inputs=layer_selector,
            outputs=meta_output,
        )

        top_k_input.change(
            _change_top_k,
            inputs=top_k_input,
            outputs=[meta_output, top_k_input],
        )

    return demo


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DETree Gradio demo.")
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--pooling", type=str, default="max")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.97)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--device", type=str)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    detector = Detector(
        database_path=args.database_path,
        model_name_or_path=args.model_name_or_path,
        pooling=args.pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        top_k=args.top_k,
        threshold=args.threshold,
        layer=args.layer,
        device=args.device,
    )

    demo = build_interface(detector)
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
