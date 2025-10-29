import torch
import cv2
import os
import random
from torch.nn.functional import softmax
from torchvision import transforms
from PIL import Image

from predictor.classifier import ResNet50Classifier


class Prediction:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract classes and model from the save model
        self.classes, self.model = self._get_model(model_path)

        # Transforms object for the data
        self.transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.486, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _get_model(self, model_path):
        """
        The function takes model_path and return the classes name (list) and model in evaluation mode
        """
        # Load the existing model dict from the directory
        checkpoint = torch.load(model_path, map_location=self.device)
        classes = checkpoint["classes"]
        model_state_dict = checkpoint["model_state_dict"]

        # load the model from the model state
        model = ResNet50Classifier(len(classes)).to(self.device)
        model.load_state_dict(model_state_dict)
        model.eval()

        return classes, model

    def __preprocess(self, image_path):
        """
        The function takes image path and open image with
        Pillow format and process the image with transforms object and return the image batch.
        """
        img = Image.open(image_path).convert("RGB")
        img = self.transforms(img).unsqueeze(0).to(self.device)

        return img

    def _predict(self, image_batch):
        """
        The function takes image_batch as a arguments and return prediction (dict) object
        """
        with torch.no_grad():
            output = self.model(image_batch)
        probabilities = softmax(output[0], dim=0)
        values, ids = torch.topk(probabilities, 5)

        prob_dict = {}
        for value, id in zip(values, ids):
            prob_dict[self.classes[id]] = value.item()

        return prob_dict

    def __modify_image(self, image_path, results):
        """
        The function puts label on a the predicted image and save to the local directory
        """
        # Get first key from the dict
        top_res = next(iter(results))

        # Read the image in BGR mode
        image = cv2.imread(image_path)
        image = cv2.resize(image, (640, 640))

        # Put label on the image with random background color
        color = (random.randint(0, 220), random.randint(0, 220), random.randint(0, 220))
        label = f"{top_res}: {results[top_res]:.2f}"

        # Create background for the lable
        text_size, _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            image,
            (image.shape[0] // 2 - 15, 0),
            (image.shape[0] // 2 - 15 + text_size[0] + 10, text_size[1] + 10),
            color,
            -1,
        )

        # Puts label text on the image
        cv2.putText(
            image,
            label,
            (image.shape[0] // 2 - 10, text_size[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        output_path = "predicted_image.jpg"
        cv2.imwrite(output_path, image)

        return output_path

    def inference(self, image_path):
        """
        The function takes image path and return a tuple (label, image)
        that is predicted using the model.
        """
        image_batch = self.__preprocess(image_path=image_path)
        results = self._predict(image_batch=image_batch)
        output_image = self.__modify_image(image_path=image_path, results=results)

        label = next(iter(results))

        return label, output_image
