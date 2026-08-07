import os

from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="ResembleAI/resemble-enhance",
    filename="enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt",
    token=os.environ.get("HF_TOKEN"),
)
print(path)
