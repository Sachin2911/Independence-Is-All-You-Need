import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE 

device = "cuda" #We will pretty much always use vast

#Load in the model
model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)

#Load one Gemma Scope SAE (layer 12 residual, 16k width)
sae, config, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device=device,
)

prompt = "When John and Mary went to the shops, John gave a drink to"
tokens = model.to_tokens(prompt)
logits, cache = model.run_with_cache(tokens)

# 4. Pull the residual-stream activation at layer 12 and encode through the SAE
resid = cache["blocks.12.hook_resid_post"]   # shape: [batch, seq, d_model]
sae_acts = sae.encode(resid)                 # shape: [batch, seq, n_features]

print(f"SAE activations shape: {sae_acts.shape}")
print(f"Active features on last token: {(sae_acts[0, -1] > 0).sum().item()}")