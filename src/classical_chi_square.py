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
)


def extract_lsb_values(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    image_array = np.array(image)

    lsb_values = image_array & 1
    lsb_values = lsb_values.flatten()

    return lsb_values


def chi_square_score(image_path):
    lsb_values = extract_lsb_values(image_path)

    zero_count = np.sum(lsb_values == 0)
    one_count = np.sum(lsb_values == 1)

    observed = np.array([zero_count, one_count])
    expected = np.array([len(lsb_values) / 2, len(lsb_values) / 2])

    chi_statistic, p_value = chisquare(f_obs=observed, f_exp=expected)

    return chi_statistic, p_value


def load_dataset():
    clean_dir = Path("data/processed/clean")
    stego_dir = Path("data/processed/stego")

    clean_images = list(clean_dir.glob("*.png"))
    stego_images = list(stego_dir.glob("*.png"))

    image_paths = clean_images + stego_images
    labels = [0] * len(clean_images) + [1] * len(stego_images)

    return image_paths, labels


def evaluate_chi_square(threshold=0.05):
    image_paths, labels = load_dataset()

    predictions = []
    scores = []

    for image_path in image_paths:
        chi_statistic, p_value = chi_square_score(image_path)

        # Lower p-value means stronger deviation from expected random LSB distribution.
        prediction = 1 if p_value < threshold else 0

        predictions.append(prediction)
        scores.append(1 - p_value)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    try:
        auc_value = roc_auc_score(labels, scores)
    except ValueError:
        auc_value = 0.0

    print("Chi-Square Statistical Steganalysis")
    print("----------------------------------")
    print(f"Threshold: {threshold}")
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
    evaluate_chi_square(threshold=0.05)