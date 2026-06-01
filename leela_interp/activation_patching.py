"""Activation patching: patch corrupted activations into clean run.

Usage:
    python leela_interp/activation_patching.py --model PATH

Metrics: KL divergence + log-odds reduction of correct move.
"""
import pickle, warnings, argparse, sys, numpy as np, torch, matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, "leela_pytorch_impl")
from model import Lc0Model
from leela_board import LeelaBoard
from utils import sq2idx

N_PUZZLES = 50
N_LAYERS = 15


def kl_divergence(p_logits, q_logits):
    p = torch.softmax(p_logits, dim=-1).clamp(min=1e-9)
    q = torch.softmax(q_logits, dim=-1).clamp(min=1e-9)
    return (p * (p.log() - q.log())).sum().item()


def log_odds_reduction(clean_logits, patched_logits, correct_idx):
    """Reduction in log-odds of the correct move after patching."""
    cp = torch.softmax(clean_logits, dim=-1)
    pp = torch.softmax(patched_logits, dim=-1)
    clean_lo = (cp[correct_idx] / (1 - cp[correct_idx] + 1e-9)).log().item()
    patched_lo = (pp[correct_idx] / (1 - pp[correct_idx] + 1e-9)).log().item()
    return clean_lo - patched_lo  # positive = patching hurt the correct move


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="leela_pytorch_impl/lc0.onnx")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-puzzles", type=int, default=N_PUZZLES)
    p.add_argument("--patch-what", default="attn_skip",
                   choices=["ffn_skip", "attn_skip", "ln2"],
                   help="Which residual stream location to patch")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading puzzles...")
    with open("leela_interp/interesting_puzzles.pkl", "rb") as f:
        puzzles = pickle.load(f)

    puzzles = puzzles[puzzles["corrupted_fen"].notna()].copy()
    print(f"Total with corrupted FEN: {len(puzzles)}")

    print(f"Loading model: {args.model}")
    model = Lc0Model(args.model, device=args.device)
    inner = model._lc0_model

    # --- Filter: only puzzles where the model actually disagrees ---
    # First, try puzzles where different_targets is True (pre-computed)
    # Then sample from those for speed
    puzzles_diff = puzzles[puzzles["different_targets"] == True]
    print(f"Puzzles with different_targets=True: {len(puzzles_diff)}")

    # Shuffle and take a candidate pool to test
    candidate = puzzles_diff.sample(n=min(500, len(puzzles_diff)), random_state=42)
    print(f"Testing KL on {len(candidate)} candidates...")
    good_puzzles = []
    for _, row in tqdm(candidate.iterrows(), total=len(candidate)):
        cb = LeelaBoard.from_fen(row["FEN"])
        crb = LeelaBoard.from_fen(row["corrupted_fen"])
        ci = model.make_inputs([cb])
        cri = model.make_inputs([crb])
        with torch.no_grad():
            co = inner(ci)
            cro = inner(cri)
        kl = kl_divergence(co[0][0], cro[0][0])
        if kl > 0.3:
            good_puzzles.append((row, kl))
        if len(good_puzzles) >= args.n_puzzles * 2:
            break

    print(f"Found {len(good_puzzles)} puzzles with KL > 0.3")
    # Take top N by KL
    good_puzzles.sort(key=lambda x: -x[1])
    selected = good_puzzles[: args.n_puzzles]
    print(f"Using top {len(selected)} (mean KL: {np.mean([s[1] for s in selected]):.4f})")

    # --- Layer names ---
    module_dict = dict(inner.named_modules())
    suffix = {"ffn_skip": "/ffn/skip", "attn_skip": "/mha/out/skip", "ln2": "/ln2"}[args.patch_what]
    all_layer_names = []
    if f"attn_body{suffix}" in module_dict:
        all_layer_names.append(f"attn_body{suffix}")
    all_layer_names += [f"encoder{i}{suffix}" for i in range(N_LAYERS)]

    # Verify all modules exist
    valid_names = []
    for name in all_layer_names:
        if name in module_dict:
            valid_names.append(name)
        else:
            print(f"  WARNING: module '{name}' not found, skipping")
            valid_names.append(None)
    N_VALID = sum(1 for n in valid_names if n is not None)
    print(f"Patching at {N_VALID} layers (patch_what={args.patch_what})")

    # --- Run patching (two conditions: move squares vs random square) ---
    results_move = {i: [] for i in range(len(valid_names))}
    results_random = {i: [] for i in range(len(valid_names))}
    # Also track log-odds reduction
    lodds_move = {i: [] for i in range(len(valid_names))}
    lodds_random = {i: [] for i in range(len(valid_names))}

    for row, base_kl in tqdm(selected, desc="Puzzles"):
        cb = LeelaBoard.from_fen(row["FEN"])
        crb = LeelaBoard.from_fen(row["corrupted_fen"])
        ci = model.make_inputs([cb])
        cri = model.make_inputs([crb])

        # Get clean top move and find source + target squares
        with torch.no_grad():
            co = inner(ci)
        clean_logits = co[0][0]
        policy, wdl, _ = model.play(cb, return_probs=True)
        top_uci = list(model.top_moves(cb, policy, top_k=1).keys())[0]
        # Paper methodology: patch only the TARGET square of the top move
        to_sq = sq2idx(top_uci[2:4], cb.pc_board.turn)
        move_squares = [to_sq]

        # Get the correct move index (model's top move on clean position)
        correct_idx = clean_logits.argmax().item()

        # Pick a random square that is NOT the target
        all_squares = list(range(64))
        candidates = [s for s in all_squares if s != to_sq]
        random_sq = [int(np.random.choice(candidates))]

        # --- Capture corrupted activations (once for all layers) ---
        corrupted_acts = {}
        hooks = []
        def make_hook(name):
            def hook(module, inp, out):
                corrupted_acts[name] = out.detach().clone()
            return hook
        for name in valid_names:
            if name is not None:
                hooks.append(module_dict[name].register_forward_hook(make_hook(name)))
        with torch.no_grad():
            _ = inner(cri)
        for h in hooks:
            h.remove()

        # --- Patch each layer with BOTH move squares and random square ---
        for layer_idx, layer_name in enumerate(valid_names):
            if layer_name is None or layer_name not in corrupted_acts:
                results_move[layer_idx].append(np.nan)
                results_random[layer_idx].append(np.nan)
                lodds_move[layer_idx].append(np.nan)
                lodds_random[layer_idx].append(np.nan)
                continue

            corr_act = corrupted_acts[layer_name]
            mod = module_dict[layer_name]

            # --- Move squares patch ---
            def make_patch_hook(corrupted, sqs):
                def hook(module, inp, out):
                    patched = out.clone()
                    patched[sqs] = corrupted[sqs]
                    return patched
                return hook

            h = mod.register_forward_hook(make_patch_hook(corr_act, move_squares))
            with torch.no_grad():
                patched_out = inner(ci)
            h.remove()
            results_move[layer_idx].append(kl_divergence(clean_logits, patched_out[0][0]))
            lodds_move[layer_idx].append(log_odds_reduction(clean_logits, patched_out[0][0], correct_idx))

            # --- Random square patch (control) ---
            h = mod.register_forward_hook(make_patch_hook(corr_act, random_sq))
            with torch.no_grad():
                patched_out = inner(ci)
            h.remove()
            results_random[layer_idx].append(kl_divergence(clean_logits, patched_out[0][0]))
            lodds_random[layer_idx].append(log_odds_reduction(clean_logits, patched_out[0][0], correct_idx))

    # --- Plot both conditions ---
    label_idx = 0
    labels = []
    move_means, move_stds = [], []
    rand_means, rand_stds = [], []
    lodd_move_means, lodd_move_stds = [], []
    lodd_rand_means, lodd_rand_stds = [], []
    for i, name in enumerate(valid_names):
        move_vals = [v for v in results_move[i] if not np.isnan(v)]
        rand_vals = [v for v in results_random[i] if not np.isnan(v)]
        lodd_mv = [v for v in lodds_move[i] if not np.isnan(v)]
        lodd_rv = [v for v in lodds_random[i] if not np.isnan(v)]
        if move_vals:
            if name is not None and "attn_body" in name:
                labels.append("embed+FFN")
            elif name is not None:
                enc_num = name.split("encoder")[1].split("/")[0]
                labels.append(f"enc {enc_num}")
            else:
                labels.append(f"layer {label_idx}")
            label_idx += 1
            move_means.append(np.mean(move_vals))
            move_stds.append(np.std(move_vals))
            rand_means.append(np.mean(rand_vals))
            rand_stds.append(np.std(rand_vals))
            lodd_move_means.append(np.mean(lodd_mv))
            lodd_move_stds.append(np.std(lodd_mv))
            lodd_rand_means.append(np.mean(lodd_rv))
            lodd_rand_stds.append(np.std(lodd_rv))

    model_name = args.model.split("/")[-1].replace(".onnx", "")
    xs = np.arange(len(move_means))

    # --- Two subplots: KL and Log-Odds Reduction ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    # === Top: KL Divergence ===
    ax1.errorbar(xs, move_means, yerr=move_stds, marker="o", capsize=4, linewidth=2,
                 markersize=6, color="#2196F3", ecolor="#90CAF9", label="Target square")
    ax1.errorbar(xs, rand_means, yerr=rand_stds, marker="s", capsize=4, linewidth=2,
                 markersize=6, color="#FF9800", ecolor="#FFCC80", linestyle="--",
                 label="Random square")
    ax1.set_ylabel("KL Divergence", fontsize=12)
    ax1.set_title(f"Activation Patching ({args.patch_what}) — {model_name}, {len(selected)} puzzles", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)

    max_kl = np.nanargmax(move_means)
    ax1.annotate(f"max {move_means[max_kl]:.3f}", xy=(max_kl, move_means[max_kl]),
                 xytext=(max_kl + 1.2, move_means[max_kl] * 1.08),
                 arrowprops=dict(arrowstyle="->", color="#2196F3"), fontsize=9, color="#2196F3")

    # === Bottom: Log-Odds Reduction ===
    ax2.errorbar(xs, lodd_move_means, yerr=lodd_move_stds, marker="o", capsize=4, linewidth=2,
                 markersize=6, color="#4CAF50", ecolor="#A5D6A7", label="Target square")
    ax2.errorbar(xs, lodd_rand_means, yerr=lodd_rand_stds, marker="s", capsize=4, linewidth=2,
                 markersize=6, color="#FF9800", ecolor="#FFCC80", linestyle="--",
                 label="Random square")
    ax2.set_ylabel("Log-Odds Reduction", fontsize=12)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    max_lodd = np.nanargmax(lodd_move_means)
    ax2.annotate(f"max {lodd_move_means[max_lodd]:.3f}", xy=(max_lodd, lodd_move_means[max_lodd]),
                 xytext=(max_lodd + 1.2, lodd_move_means[max_lodd] * 1.08),
                 arrowprops=dict(arrowstyle="->", color="#4CAF50"), fontsize=9, color="#4CAF50")

    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    out_path = f"leela_interp/activation_patching_{model_name}_with_control.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

    print(f"\n{'Layer':<12} {'KL move':>8} {'KL rand':>8} {'Δ KL':>8}  |  {'LoR move':>8} {'LoR rand':>8} {'Δ LoR':>8}")
    print("-" * 75)
    for i, label in enumerate(labels):
        print(f"{label:<12} {move_means[i]:8.4f} {rand_means[i]:8.4f} {move_means[i]-rand_means[i]:8.4f}  |  "
              f"{lodd_move_means[i]:8.4f} {lodd_rand_means[i]:8.4f} {lodd_move_means[i]-lodd_rand_means[i]:8.4f}")

    plt.show()


if __name__ == "__main__":
    main()
