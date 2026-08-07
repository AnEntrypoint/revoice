import argparse
import json

import torch
from safetensors.torch import save_file


def fuse_weight_norm(state_dict):
    fused = {}
    consumed = set()
    for key in state_dict:
        if key.endswith(".weight_g"):
            base = key[: -len(".weight_g")]
            v_key = base + ".weight_v"
            if v_key in state_dict:
                g = state_dict[key]
                v = state_dict[v_key]
                norm_v = v.norm(dim=(1, 2) if v.dim() == 3 else tuple(range(1, v.dim())), keepdim=True)
                fused[base + ".weight"] = (g * v / norm_v).contiguous()
                consumed.add(key)
                consumed.add(v_key)
    for key, tensor in state_dict.items():
        if key not in consumed:
            fused[key] = tensor.contiguous()
    return fused


def convert(checkpoint_path, output_path, manifest_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["module"] if "module" in ckpt else ckpt
    fused = fuse_weight_norm(state_dict)
    tensors = {k: v.float() for k, v in fused.items() if isinstance(v, torch.Tensor)}
    save_file(tensors, output_path)
    manifest = {k: list(v.shape) for k, v in tensors.items()}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to mp_rank_00_model_states.pt")
    parser.add_argument("output", help="output .safetensors path")
    parser.add_argument("manifest", help="output JSON key/shape manifest path")
    args = parser.parse_args()
    convert(args.checkpoint, args.output, args.manifest)
