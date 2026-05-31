import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class StegoImageDataset(Dataset):
    def __init__(self, image_paths, labels):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((128, 128))

        image_array = np.array(image)

        lsb_plane = image_array & 1
        lsb_plane = lsb_plane.astype(np.float32)
        lsb_plane = np.transpose(lsb_plane, (2, 0, 1))

        image_tensor = torch.tensor(lsb_plane, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


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
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_image_paths():
    clean_dir = Path("data/processed/clean")
    stego_dir = Path("data/processed/stego")

    clean_images = list(clean_dir.glob("*.png"))
    stego_images = list(stego_dir.glob("*.png"))

    image_paths = clean_images + stego_images
    labels = [0] * len(clean_images) + [1] * len(stego_images)

    print(f"Clean images: {len(clean_images)}")
    print(f"Stego images: {len(stego_images)}")
    print(f"Total images: {len(image_paths)}")

    return image_paths, labels


def plot_confusion_matrix(y_true, y_pred, output_path):
    matrix = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Clean", "Stego"],
    )

    display.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_scores, output_path):
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_scores)
    auc_value = roc_auc_score(y_true, y_scores)

    plt.figure()
    plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_model(experiment_name="default"):
    os.makedirs("results", exist_ok=True)
    confusion_matrix_path = f"results/confusion_matrix_{experiment_name}.png"
    roc_curve_path = f"results/roc_curve_{experiment_name}.png"
    metrics_path = f"results/metrics_{experiment_name}.txt"

    device = get_device()
    print(f"Using device: {device}")

    image_paths, labels = load_image_paths()

    _, test_paths, _, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    test_dataset = StegoImageDataset(test_paths, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load("models/simple_cnn.pth", map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            stego_scores = probabilities[:, 1]
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_scores.extend(stego_scores.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_value = roc_auc_score(y_true, y_scores)

    print("\nEvaluation Metrics")
    print("------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc_value:.4f}")

    print("\nClassification Report")
    print("---------------------")
    print(classification_report(y_true, y_pred, target_names=["Clean", "Stego"]))

    plot_confusion_matrix(y_true, y_pred, confusion_matrix_path)
    plot_roc_curve(y_true, y_scores, roc_curve_path)

    with open(metrics_path, "w", encoding="utf-8") as file:
        file.write("Evaluation Metrics\n")
        file.write("------------------\n")
        file.write(f"Accuracy:  {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall:    {recall:.4f}\n")
        file.write(f"F1-score:  {f1:.4f}\n")
        file.write(f"ROC-AUC:   {auc_value:.4f}\n")
        file.write("\nClassification Report\n")
        file.write("---------------------\n")
        file.write(classification_report(y_true, y_pred, target_names=["Clean", "Stego"]))

    print(f"Confusion matrix saved to {confusion_matrix_path}")
    print(f"ROC curve saved to {roc_curve_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained CNN steganalysis model.")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="default",
        help="Name used in output files, for example payload_1_percent_large."
    )
    args = parser.parse_args()

    evaluate_model(experiment_name=args.experiment_name)
