from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import chisquare
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def extract_lsb_plane(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(image)
    lsb_plane = image_array & 1

    return lsb_plane


def calculate_block_chi_square_score(image_path, block_size=16):
    lsb_plane = extract_lsb_plane(image_path)

    height, width, channels = lsb_plane.shape
    block_scores = []

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = lsb_plane[y:y + block_size, x:x + block_size, :]

            block_values = block.flatten()

            zero_count = np.sum(block_values == 0)
            one_count = np.sum(block_values == 1)

            observed = np.array([zero_count, one_count])
            expected = np.array([len(block_values) / 2, len(block_values) / 2])

            chi_statistic, p_value = chisquare(f_obs=observed, f_exp=expected)

            block_scores.append(1 - p_value)

    average_score = np.mean(block_scores)
    max_score = np.max(block_scores)

    final_score = (average_score + max_score) / 2

    return final_score


def load_dataset():
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

    if len(image_paths) == 0:
        raise ValueError(
            "No processed images found. Supported extensions are: .png, .jpg, .jpeg, .bmp"
        )

    return image_paths, labels


def evaluate_block_chi_square(threshold=0.5, block_size=16):
    image_paths, labels = load_dataset()

    scores = []

    for image_path in image_paths:
        score = calculate_block_chi_square_score(
            image_path=image_path,
            block_size=block_size
        )

        scores.append(score)

    scores = np.array(scores)
    labels = np.array(labels)

    false_positive_rates, true_positive_rates, thresholds = roc_curve(labels, scores)
    j_scores = true_positive_rates - false_positive_rates
    best_threshold = thresholds[np.argmax(j_scores)]

    if threshold is None:
        threshold = best_threshold

    predictions = (scores > threshold).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    auc_value = roc_auc_score(labels, scores)

    print("Block-Based Chi-Square Statistical Steganalysis")
    print("----------------------------------------------")
    print(f"Threshold: {threshold:.6f}")
    print(f"Best threshold by Youden J statistic: {best_threshold:.6f}")
    print(f"Block size: {block_size}")
    print(f"Clean score mean: {scores[labels == 0].mean():.6f}")
    print(f"Stego score mean: {scores[labels == 1].mean():.6f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc_value:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(confusion_matrix(labels, predictions))

    print("\nClassification Report")
    print("---------------------")
    print(classification_report(labels, predictions, target_names=["Clean", "Stego"], zero_division=0))


if __name__ == "__main__":
    evaluate_block_chi_square(threshold=None, block_size=16)