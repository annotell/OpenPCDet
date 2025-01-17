import torch
import torchvision
from torchvision.ops import nms

print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

# Distributed test
if torch.cuda.is_available():
    print("CUDA is available.")
    print("Torchvision NMS loaded successfully:", nms)
else:
    print("CUDA not available.")