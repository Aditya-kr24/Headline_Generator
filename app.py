"""Gradio demo: upload a photo of a public event, get a grounded headline."""
import gradio as gr

from headline_generator import HeadlineGenerator

generator = HeadlineGenerator(use_ocr=True)


def run(image):
    if image is None:
        return "", "", "", "", ""
    r = generator.generate(image)
    event = f"{r.event_type} ({r.event_confidence:.0%})" if r.event_type else f"uncertain ({r.event_confidence:.0%})"
    status = "Fallback to caption (LLM headline had unsupported facts)" if r.used_fallback else "Passed grounding check"
    return r.headline, r.caption, event, r.ocr_text or "none detected", status


demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Public event image"),
    outputs=[
        gr.Textbox(label="Generated headline"),
        gr.Textbox(label="Scene caption (BLIP)"),
        gr.Textbox(label="Event type (CLIP zero-shot)"),
        gr.Textbox(label="Visible text (OCR)"),
        gr.Textbox(label="Grounding status"),
    ],
    title="Grounded Headline Generator",
    description="Generates a headline using only what is visible in the image. No names, places, or dates unless legible in the picture.",
)

if __name__ == "__main__":
    demo.launch()
