import re
from PIL import Image
import numpy as np
import argparse


def bits_to_text(bits):
    chars = []

    for i in range(0, len(bits), 8):
        byte = bits[i:i + 8]

        if len(byte) == 8:
            chars.append(chr(int(byte, 2)))

    return "".join(chars)


def extract_text_lsb(image_path):
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    flat_pixels = image_array.reshape(-1, 3)

    bits = ""

    for pixel in flat_pixels:
        for channel in range(3):
            bits += str(pixel[channel] & 1)

    extracted_text = bits_to_text(bits)

    if "###END###" in extracted_text:
        return extracted_text.split("###END###")[0]

    return ""


def classify_payload(payload):
    payload_lower = payload.lower()

    sqli_patterns = [
        r"or\s+'?1'?\s*=\s*'?1'?",
        r"drop\s+table",
        r"union\s+select",
        r"--",
        r"select\s+.*\s+from",
    ]

    xss_patterns = [
        r"<script>",
        r"alert\s*\(",
        r"onerror\s*=",
        r"onload\s*=",
        r"javascript:",
    ]

    shell_patterns = [
        r"/bin/bash",
        r"/bin/sh",
        r"curl\s+http",
        r"wget\s+http",
        r"nc\s+-",
        r"chmod\s+\+x",
    ]

    for pattern in sqli_patterns:
        if re.search(pattern, payload_lower):
            return "SQL Injection", "High"

    for pattern in xss_patterns:
        if re.search(pattern, payload_lower):
            return "Cross-Site Scripting (XSS)", "High"

    for pattern in shell_patterns:
        if re.search(pattern, payload_lower):
            return "Shell Command / Malware-like Payload", "High"

    if payload.strip():
        return "Unknown or Benign Text", "Low"

    return "No Payload Detected", "None"


def analyze_payload(image_path):
    payload = extract_text_lsb(image_path)
    payload_type, risk_level = classify_payload(payload)

    print("Payload Analysis Result")
    print("-----------------------")
    print(f"Image path: {image_path}")
    print(f"Extracted payload: {payload}")
    print(f"Payload type: {payload_type}")
    print(f"Risk level: {risk_level}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and classify hidden payload from a stego image.")

    parser.add_argument(
        "--image",
        type=str,
        default="data/processed/stego/stego_0.png",
        help="Path of the stego image to analyze."
    )

    args = parser.parse_args()

    analyze_payload(args.image)