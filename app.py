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
    fn=displey, inputs=gr.Image(type="pil"), outputs=[gr.Text(), gr.Image()]
)

if __name__ == "__main__":
    demo.launch()
