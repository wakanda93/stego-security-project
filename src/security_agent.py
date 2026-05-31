

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from payload_classifier import extract_text_lsb, classify_payload


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def image_to_lsb_tensor(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(image)

    lsb_plane = image_array & 1
    lsb_plane = lsb_plane.astype(np.float32)
    lsb_plane = np.transpose(lsb_plane, (2, 0, 1))

    image_tensor = torch.tensor(lsb_plane, dtype=torch.float32)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def load_model(model_path, device):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def analyze_image(image_path, model_path):
    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = get_device()
    model = load_model(model_path, device)

    image_tensor = image_to_lsb_tensor(image_path).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        clean_probability = probabilities[0][0].item()
        stego_probability = probabilities[0][1].item()
        predicted_class = torch.argmax(probabilities, dim=1).item()

    prediction_label = "STEGO" if predicted_class == 1 else "CLEAN"

    print("==============================")
    print("AUTONOMOUS SECURITY AGENT")
    print("==============================")
    print(f"Image: {image_path}")
    print(f"Device: {device}")
    print()
    print("Steganalysis Result")
    print("-------------------")
    print(f"Prediction: {prediction_label}")
    print(f"Clean probability: {clean_probability * 100:.2f}%")
    print(f"Stego probability: {stego_probability * 100:.2f}%")
    print()

    if predicted_class == 1:
        payload = extract_text_lsb(str(image_path))
        payload_type, risk_level = classify_payload(payload)

        print("Payload Analysis")
        print("----------------")
        print(f"Extracted payload: {payload}")
        print(f"Payload type: {payload_type}")
        print(f"Risk level: {risk_level}")
        print()

        if risk_level == "High":
            print("Recommended action: QUARANTINE FILE")
        elif risk_level == "Low":
            print("Recommended action: REVIEW FILE MANUALLY")
        else:
            print("Recommended action: FLAG FOR FURTHER ANALYSIS")
    else:
        print("Payload Analysis")
        print("----------------")
        print("No payload extraction was performed because the image was classified as clean.")
        print()
        print("Recommended action: ALLOW FILE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous CNN-based steganalysis and payload analysis agent.")

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path of the image to analyze."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="models/simple_cnn.pth",
        help="Path of the trained CNN model."
    )

    args = parser.parse_args()

    analyze_image(
        image_path=args.image,
        model_path=args.model
    )