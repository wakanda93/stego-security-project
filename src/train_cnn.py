import numpy as np
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


class StegoImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

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


def load_image_paths():
    clean_dir = Path("data/processed/clean")
    stego_dir = Path("data/processed/stego")

    supported_extensions = {".png", ".jpg", ".jpeg", ".bmp"}

    clean_images = [
        path for path in clean_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    ]

    stego_images = [
        path for path in stego_dir.iterdir()
        if path.is_file() and path.suffix.lower() in supported_extensions
    ]

    image_paths = clean_images + stego_images
    labels = [0] * len(clean_images) + [1] * len(stego_images)

    print(f"Clean images: {len(clean_images)}")
    print(f"Stego images: {len(stego_images)}")
    print(f"Total images: {len(image_paths)}")

    if len(image_paths) == 0:
        raise ValueError(
            "No processed images found. Supported extensions are: .png, .jpg, .jpeg, .bmp"
        )

    return image_paths, labels


def train_model():
    device = get_device()
    print(f"Using device: {device}")

    image_paths, labels = load_image_paths()

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_dataset = StegoImageDataset(train_paths, train_labels)
    test_dataset = StegoImageDataset(test_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            accuracy = correct_predictions / total_samples
            progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Training Accuracy: {epoch_accuracy:.4f}"
        )

    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    test_accuracy = correct_predictions / total_samples

    print(f"Test Accuracy: {test_accuracy:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/simple_cnn.pth")

    print("Model saved to models/simple_cnn.pth")


if __name__ == "__main__":
    train_model()