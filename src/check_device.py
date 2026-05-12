import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Apple Silicon GPU aktif: MPS kullanılacak.")
else:
    device = torch.device("cpu")
    print("MPS bulunamadı. CPU kullanılacak.")

print("Device:", device)