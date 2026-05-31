import argparse
from PIL import Image
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm
import csv


def text_to_bits(text):
    return "".join(format(ord(char), "08b") for char in text)


def embed_text_lsb(image_path, output_path, text, payload_rate=0.1, output_format="png"):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    img_array = np.array(image)
    flat_pixels = img_array.reshape(-1, 3)

    max_bits = int(len(flat_pixels) * 3 * payload_rate)

    payload_text = text + "###END###"
    bits = text_to_bits(payload_text)

    while len(bits) < max_bits:
        bits += text_to_bits(text)

    bits = bits[:max_bits]

    bit_index = 0

    for i in range(len(flat_pixels)):
        for channel in range(3):
            if bit_index < len(bits):
                flat_pixels[i][channel] = (flat_pixels[i][channel] & 254) | int(bits[bit_index])
                bit_index += 1
            else:
                break

        if bit_index >= len(bits):
            break

    stego_array = flat_pixels.reshape(img_array.shape)
    stego_image = Image.fromarray(stego_array.astype(np.uint8))

    if output_format.lower() in ["jpg", "jpeg"]:
        stego_image.save(output_path, quality=95)
    else:
        stego_image.save(output_path)


def save_clean_copy(image_path, output_path, output_format="png"):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    if output_format.lower() in ["jpg", "jpeg"]:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)


def load_payloads(payload_file):
    with open(payload_file, "r", encoding="utf-8") as file:
        payloads = [line.strip() for line in file.readlines() if line.strip()]

    return payloads


def generate_dataset(payload_rate, output_format):
    raw_clean_dir = Path("data/raw_clean")
    clean_output_dir = Path("data/processed/clean")
    stego_output_dir = Path("data/processed/stego")
    payload_file = Path("payloads/payloads.txt")

    os.makedirs("results", exist_ok=True)
    payload_log_path = Path("results/payload_log.csv")

    output_format = output_format.lower()

    if output_format == "jpeg":
        output_format = "jpg"

    if output_format not in ["png", "bmp", "jpg"]:
        raise ValueError("Unsupported output format. Use png, bmp, or jpg.")

    clean_output_dir.mkdir(parents=True, exist_ok=True)
    stego_output_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_payloads(payload_file)

    image_paths = (
        list(raw_clean_dir.glob("*.png"))
        + list(raw_clean_dir.glob("*.jpg"))
        + list(raw_clean_dir.glob("*.jpeg"))
        + list(raw_clean_dir.glob("*.bmp"))
    )

    if len(image_paths) == 0:
        raise ValueError("No images found in data/raw_clean.")

    with open(payload_log_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "index",
            "source_image",
            "clean_image",
            "stego_image",
            "payload",
            "payload_rate",
            "output_format"
        ])

        for index, image_path in enumerate(tqdm(image_paths, desc="Generating dataset")):
            clean_output_path = clean_output_dir / f"clean_{index}.{output_format}"
            stego_output_path = stego_output_dir / f"stego_{index}.{output_format}"

            selected_payload = random.choice(payloads)

            save_clean_copy(
                image_path=image_path,
                output_path=clean_output_path,
                output_format=output_format
            )

            embed_text_lsb(
                image_path=image_path,
                output_path=stego_output_path,
                text=selected_payload,
                payload_rate=payload_rate,
                output_format=output_format
            )

            writer.writerow([
                index,
                str(image_path),
                str(clean_output_path),
                str(stego_output_path),
                selected_payload,
                payload_rate,
                output_format
            ])

    print("Dataset generation completed.")
    print(f"Payload rate: {payload_rate}")
    print(f"Output format: {output_format}")
    print(f"Clean images: {len(image_paths)}")
    print(f"Stego images: {len(image_paths)}")
    print(f"Payload log saved to: {payload_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clean and stego image datasets.")

    parser.add_argument(
        "--payload-rate",
        type=float,
        default=0.1,
        help="Fraction of available LSB capacity used for embedding. Example: 0.01 means 1 percent."
    )

    parser.add_argument(
        "--output-format",
        type=str,
        default="png",
        choices=["png", "bmp", "jpg", "jpeg"],
        help="Output image format for generated clean and stego samples."
    )

    args = parser.parse_args()

    generate_dataset(
        payload_rate=args.payload_rate,
        output_format=args.output_format
    )