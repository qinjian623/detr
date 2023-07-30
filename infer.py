import time

from hubconf import detr_resnet50
import torch

detr = detr_resnet50(pretrained=True)
detr.eval()

image = torch.randn((2, 3, 576, 1024))
s = time.time()
for i in range(10):
    _ = detr(image)
print(time.time() - s)

s = time.time()
for i in range(10):
    _ = detr(image)
print(time.time() - s)

