"""
Activation patching: patch corrupted activations into clean run, measure KL.

Usage:
    python leela_interp/activation_patching.py [MODEL_PATH]

Finds the model's top move on a clean position, then patches corrupted
activations at the source + target squares for each layer.
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="leela_pytorch_impl/lc0-original.onnx")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n-puzzles", type=int, default=N_PUZZLES)
    p.add_argument("--patch-what", default="ffn_skip",
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

    # --- Run patching ---
    results = {i: [] for i in range(len(valid_names))}

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
        from_sq = sq2idx(top_uci[:2], cb.pc_board.turn)
        to_sq = sq2idx(top_uci[2:4], cb.pc_board.turn)
        patch_squares = [from_sq, to_sq]  # patch both source and target

        # --- Capture corrupted activations ---
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

        # --- Patch each layer ---
        for layer_idx, layer_name in enumerate(valid_names):
            if layer_name is None or layer_name not in corrupted_acts:
                results[layer_idx].append(np.nan)
                continue

            corr_act = corrupted_acts[layer_name]  # (1*64, D)
            squares_local = patch_squares

            def make_patch_hook(corrupted, sqs):
                def hook(module, inp, out):
                    patched = out.clone()
                    patched[sqs] = corrupted[sqs]
                    return patched
                return hook

            mod = module_dict[layer_name]
            h = mod.register_forward_hook(make_patch_hook(corr_act, squares_local))
            with torch.no_grad():
                patched_out = inner(ci)
            h.remove()

            kl = kl_divergence(clean_logits, patched_out[0][0])
            results[layer_idx].append(kl)

    # --- Plot ---
    # Build labels dynamically based on what we actually patched
    label_idx = 0
    labels, means_list, stds_list = [], [], []
    for i, name in enumerate(valid_names):
        vals = [v for v in results[i] if not np.isnan(v)]
        if vals:
            if name is not None and "attn_body" in name:
                labels.append("embed+FFN")
            elif name is not None:
                enc_num = name.split("encoder")[1].split("/")[0]
                labels.append(f"enc {enc_num}")
            else:
                labels.append(f"layer {label_idx}")
            label_idx += 1
            means_list.append(np.mean(vals))
            stds_list.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(12, 5))
    xs = np.arange(len(means_list))
    ax.errorbar(xs, means_list, yerr=stds_list, marker="o", capsize=4, linewidth=2,
                markersize=6, color="#2196F3", ecolor="#90CAF9")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("KL Divergence (clean vs patched policy)", fontsize=12)
    ax.set_xlabel("Layer", fontsize=12)
    model_name = args.model.split("/")[-1].replace(".onnx", "")
    ax.set_title(
        f"Activation Patching: patch source+target squares ({args.patch_what})\n"
        f"{model_name}, {len(selected)} puzzles, error bars = ±1 std",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    max_idx = np.nanargmax(means_list)
    ax.annotate(
        f"Max KL = {means_list[max_idx]:.4f}",
        xy=(max_idx, means_list[max_idx]),
        xytext=(max_idx + 1.5, means_list[max_idx] * 1.05),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=10, color="red",
    )

    plt.tight_layout()
    out_path = f"leela_interp/activation_patching_{model_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")

    # Print top layers
    print("\nTop layers by KL divergence:")
    sorted_layers = sorted(zip(labels, means_list), key=lambda x: -x[1])
    for name, kl in sorted_layers[:8]:
        print(f"  {name}: {kl:.6f}")

    plt.show()


if __name__ == "__main__":
    main()
