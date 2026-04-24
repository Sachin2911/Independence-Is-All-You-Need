"""
Pilot pipeline: Gemma-2-2B -> Gemma Scope SAE -> PC algorithm -> DAG viz.

This extends the baseline to:
  1. Collect SAE activations over a batch of prompts
  2. Select the top-K highest-variance features
  3. Binarize and run the PC algorithm
  4. Visualize the learned CPDAG
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import csv
import html
import json
import re
import urllib.request
from pathlib import Path
from transformer_lens import HookedTransformer
from sae_lens import SAE

# causal-learn is the right library for PC
# pip install causal-learn
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

device = "cuda"
plots_dir = Path("plots")
plots_dir.mkdir(exist_ok=True)

LAYER = 12
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
N_FEATURES_TO_LEARN = 20   # keep small for pilot — PC scales badly with #vars
N_PROMPTS = 500             # real data collection will be ~4000


def _clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_positive_logits(html_text: str, max_items: int = 5) -> list[str]:
    """Extract a few top positive-logit tokens as fallback metadata."""
    section_match = re.search(
        r"POSITIVE LOGITS(.*?)(?:Activations Density|# No Known Activations|Test|Steer)",
        html_text,
        flags=re.S,
    )
    if not section_match:
        return []
    section = section_match.group(1)
    tokens = re.findall(r"```\s*(.*?)\s*```", section)

    cleaned = []
    for tok in tokens:
        tok = _clean_text(tok)
        if tok:
            cleaned.append(tok)
        if len(cleaned) >= max_items:
            break
    return cleaned


def fetch_neuronpedia_metadata(feature_id: int) -> dict:
    """Fetch feature metadata from Neuronpedia feature API."""
    url = f"https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/{feature_id}"
    api_url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/12-gemmascope-res-16k/{feature_id}"
    result = {
        "feature_id": feature_id,
        "url": url,
        "api_url": api_url,
        "status": "ok",
        "title": "",
        "explanation": "unavailable",
        "positive_logits_hint": "",
        "error": "",
        "raw_html": "",
    }

    try:
        req = urllib.request.Request(
            api_url,
            headers={"Accept": "application/json", "User-Agent": "python-urllib"},
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            feature_obj = json.loads(response.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

    result["title"] = (
        f"{feature_obj.get('modelId', 'gemma-2-2b')} · "
        f"{feature_obj.get('layer', '12-gemmascope-res-16k')} · "
        f"{feature_obj.get('index', feature_id)}"
    )

    explanations = feature_obj.get("explanations", [])
    if isinstance(explanations, list):
        for exp in explanations:
            if not isinstance(exp, dict):
                continue
            desc = _clean_text(str(exp.get("description", "")))
            if desc:
                result["explanation"] = desc
                break

    pos_tokens = feature_obj.get("pos_str", [])
    if isinstance(pos_tokens, list):
        cleaned = []
        for tok in pos_tokens:
            if isinstance(tok, str):
                t = _clean_text(tok)
                if t:
                    cleaned.append(t)
            if len(cleaned) >= 5:
                break
        if cleaned:
            result["positive_logits_hint"] = ", ".join(cleaned)

    if result["explanation"] == "unavailable" and result["positive_logits_hint"]:
        result["explanation"] = f"logit tokens: {result['positive_logits_hint']}"

    return result


# ---------------------------------------------------------------------------
# Load model + SAE
# ---------------------------------------------------------------------------

model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)
model.eval()

sae, _, _ = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id=f"layer_{LAYER}/width_16k/canonical",
    device=device,
)
sae.eval()

# ---------------------------------------------------------------------------
# Generate a simple IOI-style prompt set
# ---------------------------------------------------------------------------
# For the pilot we just want variety — real experiments use proper IOI templates.

NAMES = ["John", "Mary", "Alice", "Bob", "Tom", "Sarah", "David", "Emma",
         "Michael", "Lisa", "James", "Anna", "Peter", "Kate", "Sam", "Rose"]
PLACES = ["shops", "park", "cafe", "library", "gym", "beach", "market", "museum"]
OBJECTS = ["drink", "book", "ball", "gift", "letter", "phone", "bag", "key"]

def make_ioi_prompts(n):
    rng = np.random.default_rng(0)
    prompts = []
    for _ in range(n):
        a, b = rng.choice(NAMES, size=2, replace=False)
        place = rng.choice(PLACES)
        obj = rng.choice(OBJECTS)
        prompts.append(f"When {a} and {b} went to the {place}, {a} gave a {obj} to")
    return prompts

prompts = make_ioi_prompts(N_PROMPTS)
print(f"Generated {len(prompts)} IOI prompts")

# ---------------------------------------------------------------------------
# Collect SAE activations at the last token of each prompt
# ---------------------------------------------------------------------------

all_activations = []
with torch.inference_mode():
    for i, prompt in enumerate(prompts):
        if i % 50 == 0:
            print(f"  processing prompt {i}/{len(prompts)}")
        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name == HOOK_NAME,
            stop_at_layer=LAYER + 1,
            return_type=None,
        )
        resid = cache[HOOK_NAME]                 # [1, seq, d_model]
        last_token_resid = resid[0, -1]          # [d_model]
        feats = sae.encode(last_token_resid)     # [n_features]
        all_activations.append(feats.detach().float().cpu().numpy())

X = np.stack(all_activations)   # [n_prompts, n_features]
print(f"Activation matrix shape: {X.shape}")

# ---------------------------------------------------------------------------
# Feature selection: top-K by variance
# ---------------------------------------------------------------------------

variances = X.var(axis=0)
top_feature_idx = np.argsort(variances)[::-1][:N_FEATURES_TO_LEARN]
X_selected = X[:, top_feature_idx]
print(f"Selected {N_FEATURES_TO_LEARN} features by variance. Indices: {top_feature_idx.tolist()}")

# ---------------------------------------------------------------------------
# Binarize (active vs inactive) — required for chi-squared CI test
# ---------------------------------------------------------------------------

X_binary = (X_selected > 0).astype(int)
print(f"Active rate per feature:")
for i, idx in enumerate(top_feature_idx):
    rate = X_binary[:, i].mean()
    print(f"  feature {idx}: {rate:.2%} active")

# Drop features that are always-on or always-off — PC cannot learn from them
keep_mask = (X_binary.mean(axis=0) > 0.05) & (X_binary.mean(axis=0) < 0.95)
X_binary = X_binary[:, keep_mask]
kept_feature_idx = top_feature_idx[keep_mask]
print(f"After filtering: {X_binary.shape[1]} usable features")

# ---------------------------------------------------------------------------
# Run PC
# ---------------------------------------------------------------------------

print("Running PC algorithm with chi-squared CI test...")
cg = pc(
    X_binary,
    alpha=0.05,
    indep_test=chisq,
    show_progress=False,
)
# cg.G is a GeneralGraph; cg.G.graph is an adjacency matrix where
#   graph[i, j] = 1, graph[j, i] = -1  means  i -> j
#   graph[i, j] = -1, graph[j, i] = -1 means  i -- j  (undirected)

adj = cg.G.graph
n = adj.shape[0]
G_dir = nx.DiGraph()
G_undir_edges = []

# Use feature IDs as labels so the graph is interpretable
labels = {i: f"f{kept_feature_idx[i]}" for i in range(n)}
G_dir.add_nodes_from(range(n))

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if adj[i, j] == 1 and adj[j, i] == -1:   # i -> j
            G_dir.add_edge(i, j)
        elif adj[i, j] == -1 and adj[j, i] == -1 and i < j:   # undirected, add once
            G_undir_edges.append((i, j))

print(f"Learned graph: {G_dir.number_of_edges()} directed edges, "
      f"{len(G_undir_edges)} undirected edges")

# ---------------------------------------------------------------------------
# Fetch Neuronpedia metadata + write CSV
# ---------------------------------------------------------------------------

feature_metadata = {}
for idx in kept_feature_idx:
    feature_metadata[int(idx)] = fetch_neuronpedia_metadata(int(idx))

raw_metadata_path = plots_dir / "feature_metadata_raw.json"
with raw_metadata_path.open("w", encoding="utf-8") as f:
    json.dump(
        {str(k): v for k, v in feature_metadata.items()},
        f,
        ensure_ascii=False,
        indent=2,
    )

metadata_path = plots_dir / "feature_metadata.csv"
with metadata_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "feature_id",
            "active_rate",
            "explanation",
            "positive_logits_hint",
            "status",
            "url",
        ]
    )
    for col, feature_id in enumerate(kept_feature_idx):
        meta = feature_metadata[int(feature_id)]
        writer.writerow([
            int(feature_id),
            float(X_binary[:, col].mean()),
            meta["explanation"],
            meta["positive_logits_hint"],
            meta["status"],
            meta["url"],
        ])

# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G_dir, seed=42, k=1.5)
G_undir = nx.Graph()
G_undir.add_nodes_from(G_dir.nodes())
G_undir.add_edges_from(G_undir_edges)

# Highlight a few structurally important nodes with explanation snippets.
node_scores = {node: G_dir.degree(node) + G_undir.degree(node) for node in G_dir.nodes()}
top_nodes = sorted(node_scores, key=node_scores.get, reverse=True)[:5]
for node in top_nodes:
    feature_id = int(kept_feature_idx[node])
    expl = feature_metadata.get(feature_id, {}).get("explanation", "unavailable")
    short_expl = expl[:55] + "..." if len(expl) > 58 else expl
    labels[node] = f"f{feature_id}\n{short_expl}"

# Directed edges
nx.draw_networkx_edges(
    G_dir, pos, edge_color="#2c3e50", arrows=True,
    arrowsize=20, width=1.5, connectionstyle="arc3,rad=0.1",
)
# Undirected edges — draw as dashed grey lines
nx.draw_networkx_edges(
    G_undir, pos, edge_color="#95a5a6", style="dashed", width=1.2,
)

nx.draw_networkx_nodes(G_dir, pos, node_color="#3498db",
                        node_size=1200, alpha=0.9)
nx.draw_networkx_labels(G_dir, pos, labels=labels, font_size=9,
                         font_color="#111111", font_weight="bold",
                         bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.2))

plt.title(f"Learned Bayesian Network over SAE Features (Layer {LAYER})\n"
          f"Solid arrow: directed edge. Dashed line: undirected (Markov-equivalent).",
          fontsize=11)
plt.axis("off")
plt.tight_layout()
graph_path = plots_dir / "learned_bn.png"
plt.savefig(graph_path, dpi=180, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Look for v-structures — your headline experiment
# ---------------------------------------------------------------------------

v_structures = []
for node in G_dir.nodes():
    parents = list(G_dir.predecessors(node))
    for i, p1 in enumerate(parents):
        for p2 in parents[i+1:]:
            # v-structure: p1 -> node <- p2, and no edge between p1 and p2
            if not G_dir.has_edge(p1, p2) and not G_dir.has_edge(p2, p1):
                v_structures.append((p1, node, p2))

print(f"\nFound {len(v_structures)} v-structures (explaining-away candidates):")
for p1, c, p2 in v_structures:
    print(f"  f{kept_feature_idx[p1]} -> f{kept_feature_idx[c]} <- f{kept_feature_idx[p2]}")

print(f"\nSaved graph to: {graph_path.resolve()}")
print(f"Saved feature metadata to: {metadata_path.resolve()}")
print(f"Saved raw metadata to: {raw_metadata_path.resolve()}")
print(f"Look up feature IDs on Neuronpedia:")
print(f"  https://neuronpedia.org/gemma-2-2b/{LAYER}-gemmascope-res-16k/<feature_id>")