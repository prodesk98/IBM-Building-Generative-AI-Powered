import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray) -> str:
    raw_image = Image.fromarray(input_image).convert('RGB')
    text = "the image of"
    inputs = processor(images=raw_image, text=text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    return processor.decode(outputs[0], skip_special_tokens=True)

iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Captioning Photos with Generative AI",
    description="This app generates captions for photos using a pre-trained BLIP model.",
)

if __name__ == "__main__":
    iface.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
