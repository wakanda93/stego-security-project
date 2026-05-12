# CNN-Based Autonomous Steganalysis and Malicious Payload Detection

This project is developed for the Computer and Network Security course.

## Project Overview

The aim of this project is to detect hidden malicious-looking payloads inside image files using steganalysis techniques and deep learning.

The project contains two main phases:

1. **Attack Simulation**
   - Payloads such as SQL Injection, XSS, and shell-like commands are embedded into image files using Least Significant Bit (LSB) steganography.

2. **Defense System**
   - A CNN-based model is trained to classify images as either clean or stego.
   - Evaluation metrics such as confusion matrix, ROC curve, AUC, precision, recall, and F1-score are used.

## Current Features

- LSB payload embedding
- LSB payload extraction
- Automatic clean/stego dataset generation
- Apple Silicon MPS support for PyTorch training

## Project Structure

```text
data/
payloads/
src/
models/
results/
report/