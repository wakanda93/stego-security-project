from PIL import Image
import numpy as np


def bits_to_text(bits):
    chars = []

    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]

        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))

    return "".join(chars)


def extract_text_lsb(image_path):
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    flat_pixels = img_array.reshape(-1, 3)

    bits = ""

    for pixel in flat_pixels:
        for channel in range(3):
            bits += str(pixel[channel] & 1)

    extracted_text = bits_to_text(bits)

    if "###END###" in extracted_text:
        return extracted_text.split("###END###")[0]

    return extracted_text


if __name__ == "__main__":
    payload = extract_text_lsb("data/processed/stego/sample_stego.png")
    print("Extracted payload:", payload)