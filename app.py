import gradio as gr
from PIL import Image

from predictor.infer import Prediction

model_path = "./models/resnet50flower-e100.pth"
prediction = Prediction(model_path=model_path)


def displey(image: Image.Image):
    image_path = "input.jpg"
    image.save(image_path)

    output_image, label = prediction.inference(image_path=image_path)
    return output_image, label


demo = gr.Interface(
    fn=displey,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Text(label="Class of Image"), gr.Image(label="I")],
    title="Flower Classification",
    description="The model can classify 5 classes (Daisy, Rose, Tulip, Dandelion, Sunflower)",
)

if __name__ == "__main__":
    demo.launch()
