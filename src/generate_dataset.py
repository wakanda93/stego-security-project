from PIL import Image
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm


def text_to_bits(text):
    return "".join(format(ord(char), "08b") for char in text)


def embed_text_lsb(image_path, output_path, text, payload_rate=0.1):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))

    img_array = np.array(image)
    flat_pixels = img_array.reshape(-1, 3)

    text = text + "###END###"
    bits = text_to_bits(text)

    max_bits = int(len(flat_pixels) * 3 * payload_rate)

    if len(bits) > max_bits:
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
    stego_image.save(output_path)


def save_clean_copy(image_path, output_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128))
    image.save(output_path)


def load_payloads(payload_file):
    with open(payload_file, "r", encoding="utf-8") as file:
        payloads = [line.strip() for line in file.readlines() if line.strip()]

    return payloads


def generate_dataset():
    raw_clean_dir = Path("data/raw_clean")
    clean_output_dir = Path("data/processed/clean")
    stego_output_dir = Path("data/processed/stego")
    payload_file = Path("payloads/payloads.txt")

    clean_output_dir.mkdir(parents=True, exist_ok=True)
    stego_output_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_payloads(payload_file)

    image_paths = list(raw_clean_dir.glob("*.png")) + list(raw_clean_dir.glob("*.jpg")) + list(raw_clean_dir.glob("*.jpeg"))

    if len(image_paths) == 0:
        raise ValueError("No images found in data/raw_clean.")

    for index, image_path in enumerate(tqdm(image_paths, desc="Generating dataset")):
        clean_output_path = clean_output_dir / f"clean_{index}.png"
        stego_output_path = stego_output_dir / f"stego_{index}.png"

        selected_payload = random.choice(payloads)

        save_clean_copy(
            image_path=image_path,
            output_path=clean_output_path
        )

        embed_text_lsb(
            image_path=image_path,
            output_path=stego_output_path,
            text=selected_payload,
            payload_rate=0.1
        )

    print("Dataset generation completed.")
    print(f"Clean images: {len(image_paths)}")
    print(f"Stego images: {len(image_paths)}")


if __name__ == "__main__":
    generate_dataset()