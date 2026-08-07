import torch
from safetensors.torch import save_file

manifest = {}
with open("weights/enhancer_stage2.manifest.json") as f:
    import json

    manifest = json.load(f)

tensors = {k: torch.randn(shape) * 0.01 for k, shape in manifest.items()}
save_file(tensors, "web/test_weights.safetensors")
print(len(tensors), "tensors written")
