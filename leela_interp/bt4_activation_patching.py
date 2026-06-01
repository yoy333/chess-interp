"""
Activation patching on BT4 model using nnsight.

For each puzzle with a corrupted position:
1. Find the model's top move on the clean position and its target square
2. For each encoder layer, patch the residual stream activation at that square
   from the corrupted run into the clean run
3. Measure KL divergence between the clean policy and the patched policy
4. Plot KL divergence per layer

Based on the Leela Interp paper methodology.
"""
import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Silence the onnx2torch slice warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-tuple sequence.*")

import sys
sys.path.insert(0, "leela_pytorch_impl")
from model import Lc0Model
from leela_board import LeelaBoard
from uci_to_idx import idx_to_uci as _idx_to_uci
from utils import sq2idx

DEVICE = "cpu"
N_PUZZLES = 30  # number of puzzles to use
N_LAYERS = 15
D_MODEL = 1024
N_SQUARES = 64


def uci_to_target_square(uci: str, turn: bool) -> int:
    """Get the 0-63 index of the destination square of a UCI move."""
    to_square = uci[2:4]
    return sq2idx(to_square, turn)


def kl_divergence(p_logits, q_logits, legal_mask=None):
    """KL divergence D_KL(P || Q) between two policies in logit space."""
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    # Clamp to avoid log(0)
    p = p.clamp(min=1e-9)
    q = q.clamp(min=1e-9)
    if legal_mask is not None:
        # Only consider legal moves
        p = p[legal_mask]
        q = q[legal_mask]
        p = p / p.sum()
        q = q / q.sum()
    return (p * (p.log() - q.log())).sum().item()


def main():
    print("Loading puzzles...")
    with open("leela_interp/interesting_puzzles.pkl", "rb") as f:
        puzzles = pickle.load(f)

    # Filter for puzzles that have a corrupted FEN
    puzzles = puzzles[puzzles["corrupted_fen"].notna()].copy()
    print(f"Total puzzles with corrupted FEN: {len(puzzles)}")

    # Pick a diverse subset
    puzzles = puzzles.sample(n=min(N_PUZZLES, len(puzzles)), random_state=42)
    print(f"Using {len(puzzles)} puzzles")

    print("Loading BT4 model...")
    model = Lc0Model(
        "leela_pytorch_impl/BT4-tf13tune.onnx",
        device=DEVICE,
    )
    inner = model._lc0_model

    # Layer names for residual stream (post-FFN skip, before ln2)
    layer_names = [f"encoder{i}/ffn/skip" for i in range(N_LAYERS)]
    # Also include the initial embedding + FFN: attn_body/ffn/skip
    all_layer_names = ["attn_body/ffn/skip"] + layer_names

    print(f"Patching at layers: {all_layer_names[:3]}...{all_layer_names[-2:]}")

    # Storage for results: {layer_idx: [kl_divergences]}
    results = {i: [] for i in range(N_LAYERS + 1)}  # +1 for attn_body

    for puzzle_idx, (_, puzzle) in enumerate(tqdm(puzzles.iterrows(), desc="Puzzles")):
        # --- Clean position ---
        clean_board = LeelaBoard.from_fen(puzzle["FEN"])
        clean_input = model.make_inputs([clean_board])

        # --- Corrupted position ---
        corrupted_board = LeelaBoard.from_fen(puzzle["corrupted_fen"])
        corrupted_input = model.make_inputs([corrupted_board])

        # --- Get model's top move on clean position ---
        with torch.no_grad():
            clean_output = inner(clean_input)
        clean_logits = clean_output[0][0]  # shape (1858,)

        # Find the top legal move
        legal_indices, legal_uci = model.legal_moves(clean_board)
        legal_mask = torch.zeros(1858, dtype=torch.bool)
        legal_mask[legal_indices] = True
        top_legal_idx = clean_logits[legal_mask].argmax().item()
        top_move_uci = legal_uci[top_legal_idx]
        target_sq = uci_to_target_square(top_move_uci, clean_board.pc_board.turn)

        # Get legal move mask for KL
        legal_mask_full = legal_mask.clone()

        # --- Run corrupted forward to capture corrupted activations ---
        corrupted_acts = {}
        def make_corrupted_hook(name):
            def hook(module, inp, out):
                corrupted_acts[name] = out.detach().clone()
            return hook

        corrupted_hooks = []
        for name in all_layer_names:
            module = dict(inner.named_modules()).get(name)
            if module is not None:
                h = module.register_forward_hook(make_corrupted_hook(name))
                corrupted_hooks.append(h)

        with torch.no_grad():
            _ = inner(corrupted_input)

        for h in corrupted_hooks:
            h.remove()

        # --- For each layer, do activation patching ---
        for layer_idx, layer_name in enumerate(all_layer_names):
            module = dict(inner.named_modules()).get(layer_name)
            if module is None or layer_name not in corrupted_acts:
                results[layer_idx].append(np.nan)
                continue

            corrupted_act = corrupted_acts[layer_name]  # shape (1*64, 1024)

            # Create patching hook
            patched_output = None
            def make_patch_hook(corrupted, target_sq_local):
                def hook(module, inp, out):
                    # out shape: (batch*squares, d_model)
                    patched = out.clone()
                    patched[target_sq_local] = corrupted[target_sq_local]
                    return patched
                return hook

            patch_hook = module.register_forward_hook(
                make_patch_hook(corrupted_act, target_sq)
            )

            with torch.no_grad():
                patched_output = inner(clean_input)

            patch_hook.remove()

            patched_logits = patched_output[0][0]
            kl = kl_divergence(clean_logits, patched_logits, legal_mask_full)
            results[layer_idx].append(kl)

    # --- Aggregate and plot ---
    print("\nAggregating results...")
    layer_labels = ["embed+FFN"] + [f"encoder {i}" for i in range(N_LAYERS)]
    means = []
    stds = []
    for i in range(N_LAYERS + 1):
        vals = [v for v in results[i] if not np.isnan(v)]
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals) if vals else np.nan)

    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(means))
    ax.errorbar(xs, means, yerr=stds, marker="o", capsize=4, linewidth=2,
                markersize=6, color="#2196F3", ecolor="#90CAF9")
    ax.set_xticks(xs)
    ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("KL Divergence (clean vs patched policy)", fontsize=12)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_title(
        f"Activation Patching: Patch target square from corrupted → clean\n"
        f"BT4 model, {N_PUZZLES} puzzles, error bars = ±1 std",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Highlight the layer with max KL
    max_idx = np.nanargmax(means)
    ax.annotate(
        f"Max KL = {means[max_idx]:.4f}\nat {layer_labels[max_idx]}",
        xy=(max_idx, means[max_idx]),
        xytext=(max_idx + 1.5, means[max_idx] * 1.1),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10,
        color="red",
    )

    plt.tight_layout()
    out_path = "leela_interp/bt4_activation_patching.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

    # Print top layers
    print("\nTop layers by KL divergence:")
    sorted_layers = sorted(
        [(layer_labels[i], means[i]) for i in range(len(means)) if not np.isnan(means[i])],
        key=lambda x: -x[1],
    )
    for name, kl in sorted_layers[:5]:
        print(f"  {name}: {kl:.6f}")

    plt.show()


if __name__ == "__main__":
    main()
