import torch
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from transformer_lens import HookedTransformer
from sae_lens import SAE

device = "cuda"  # We will pretty much always use vast
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)
enriched_metadata_csv = plots_dir / "feature_metadata_enriched.csv"

# Load in the model
model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
model.eval()

# Load one Gemma Scope SAE (layer 12 residual, 16k width)
sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_12/width_16k/canonical",
    device=device,
)
sae.eval()

prompt = "When John and Mary went to the shops, John gave a drink to"
tokens = model.to_tokens(prompt)
hook_name = "blocks.12.hook_resid_post"

with torch.inference_mode():
    _, cache = model.run_with_cache(
        tokens,
        names_filter=lambda name: name == hook_name,
        stop_at_layer=13,
        return_type=None,
    )

# 4. Pull the residual-stream activation at layer 12 and encode through the SAE
resid = cache[hook_name]  # shape: [batch, seq, d_model]
sae_acts = sae.encode(resid)  # shape: [batch, seq, n_features]
sae_acts_cpu = sae_acts[0].detach().float().cpu()  # [seq, n_features]
token_labels = model.to_str_tokens(tokens)
if isinstance(token_labels[0], list):
    token_labels = token_labels[0]

print(f"SAE activations shape: {sae_acts.shape}")
print(f"Active features on last token: {(sae_acts[0, -1] > 0).sum().item()}")

# 1) Plot active feature count per token.
active_counts = (sae_acts_cpu > 0).sum(dim=-1).numpy()
plt.figure(figsize=(12, 4))
plt.plot(range(len(token_labels)), active_counts, marker="o")
plt.xticks(range(len(token_labels)), token_labels, rotation=45, ha="right")
plt.ylabel("Active SAE features (> 0)")
plt.xlabel("Token position")
plt.title("SAE Sparsity Across Tokens")
plt.tight_layout()
active_counts_path = plots_dir / "active_features_per_token.png"
plt.savefig(active_counts_path, dpi=180)
plt.close()

# 2) Plot top activated features on the last token.
last_token_acts = sae_acts_cpu[-1]
topk = 20
top_vals, top_idx = torch.topk(last_token_acts, k=topk)
top_vals_np = top_vals.numpy()
top_idx_np = top_idx.numpy()

plt.figure(figsize=(12, 5))
plt.bar(range(topk), top_vals_np)
plt.xticks(range(topk), [str(i) for i in top_idx_np], rotation=45, ha="right")
plt.ylabel("Activation value")
plt.xlabel("Feature id")
plt.title(f"Top {topk} SAE Features on Last Token")
plt.tight_layout()
topk_path = plots_dir / "top_features_last_token.png"
plt.savefig(topk_path, dpi=180)
plt.close()

# 2b) Optional metadata lookup for top features from scrap.py output.
feature_notes = {}
if enriched_metadata_csv.exists():
    with enriched_metadata_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row.get("feature_id", "").strip()
            if not fid.isdigit():
                continue
            note = row.get("explanation", "").strip() or "unavailable"
            feature_notes[int(fid)] = note

# 3) Heatmap of top last-token features across the whole sequence.
heatmap_data = sae_acts_cpu[:, top_idx_np].numpy()
plt.figure(figsize=(12, 6))
im = plt.imshow(heatmap_data.T, aspect="auto", interpolation="nearest")
plt.colorbar(im, label="Activation")
plt.yticks(range(topk), [str(i) for i in top_idx_np])
plt.xticks(range(len(token_labels)), token_labels, rotation=45, ha="right")
plt.ylabel("Feature id (top on last token)")
plt.xlabel("Token position")
plt.title("How Top Last-Token Features Behave Across Tokens")
plt.tight_layout()
heatmap_path = plots_dir / "top_features_heatmap.png"
plt.savefig(heatmap_path, dpi=180)
plt.close()

print(f"Saved plots to: {plots_dir.resolve()}")
print(f"- {active_counts_path.name}")
print(f"- {topk_path.name}")
print(f"- {heatmap_path.name}")
if feature_notes:
    print("Top feature explanations (from feature_metadata_enriched.csv):")
    for fid in top_idx_np[:10]:
        print(f"- f{int(fid)}: {feature_notes.get(int(fid), 'unavailable')}")
else:
    print("No enriched metadata found. Run `uv run python scrap.py` first.")