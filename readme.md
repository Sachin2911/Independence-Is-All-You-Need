# Independence Is All You Need

Video walkthrough: <https://www.youtube.com/watch?v=iKB93PyhBjg&t=1s>

> Constraint-based causal discovery (the PC algo) over SAE features in Gemma-2-2B on the Indirect Object Identification (IOI) task.

Attribution patching (AP), the current state-of-the-art for SAE feature-circuit recovery, is a marginal-effect estimator. As a result it cannot distinguish a chain $A \to C \to B$ from a v-structure $A \to C \leftarrow B$, because that distinction lives in conditional dependencies. The PC algorithm from PGMs closes the gap. Applied to SAE features from Gemma-2-2B on IOI, PC recovers 599 v-structures across $N = 5{,}000$ prompts, of which a sampled cross-layer subset is 100% interventionally validated.

Full paper: [`paper/Independence_Is_All_You_Need.pdf`](paper/Independence_Is_All_You_Need.pdf).

## Repository layout

```
.
├── experiments/         Python environment and standalone scripts
│   ├── Makefile         setup / install / notebook / clean targets
│   ├── pyproject.toml   uv-managed dependencies
│   ├── helpers/         shared utilities (IOI dataset generator)
│   ├── main.py          standalone entry script
│   └── plots/           pilot plots and feature metadata
├── notebooks/           experimental pipeline (run in order)
├── data/                saved tensors, CSVs, JSON outputs
├── figures/             plots used in the paper
├── paper/               LaTeX source, compiled PDF, markdown drafts
└── readme.md
```

## Notebooks

The five notebooks reproduce the full pipeline. You can run in this order; each writes the artifacts the next consumes.

1. **`IOI_Dataset_Setup.ipynb`** — Builds the 5,000-prompt IOI dataset from the templates and name pool of Wang et al. (2022).

2. **`Attribution_Patching.ipynb`** — Computes residual-stream attribution patching across all 26 layers of Gemma-2-2B at every token position, batched 64 prompts at a time.

3. **`PC_Bayesian_Net.ipynb`** — Extracts the top-40-by-activation-variance SAE features per layer at layers {18, 22, 25} from the Gemma Scope SAEs (115 features after dropping constants), binarises at threshold > 0, then runs the PC algorithm with $\chi^2$ CI tests at $\alpha = 0.01$ under the shallow → deep layer prior.

4. **`AP_PC_Comparison.ipynb`** — (a) Computes per-feature AP scores at the END token over the same 115 features and shows that AP is essentially silent in layers 18–22;  selects the cleanest case-study v-structure with Neuronpedia auto-interp labels and verifies the explaining-away signature (marginal $p = 0.603$, conditional $p < 0.0001$); validates 50 sampled cross-layer edges via $do(\cdot)$-ablation (100% pass rate, mean $|\Delta| = 5.92$). 

5. **`Robustness.ipynb`** — Re-runs PC across $\alpha \in \{0.001, 0.005, 0.01, 0.05, 0.1\}$ (two orders of magnitude) and tracks edge counts, v-structure counts, and case-study persistence.

## Quick start

```bash
git clone https://github.com/Sachin2911/Independence-Is-All-You-Need.git
cd Independence-Is-All-You-Need/experiments
make setup
make notebook
```

`make setup` uses [`uv`](https://github.com/astral-sh/uv) to install Python 3.13 and all dependencies, registers an IPython kernel called `experiments-uv`, configures GitHub auth via `gh`, and prompts for a Hugging Face token.

A read-only Hugging Face token is required to download Gemma-2-2B and the Gemma Scope SAEs. Create one at <https://huggingface.co/settings/tokens>. **Never commit personal tokens.**

`make notebook` launches JupyterLab on `0.0.0.0:8888`. Open the `notebooks/` directory and run the five notebooks in the order listed above.

## Author

Sachin Mohan · University of the Witwatersrand · <sachin@opencolab.dev>
