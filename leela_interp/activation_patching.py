"""Activation patching: patch corrupted activations into clean run.

Usage:
    python leela_interp/activation_patching.py --model PATH
    python leela_interp/activation_patching.py --model PATH --heatmap
    python leela_interp/activation_patching.py --model PATH --heatmap --heatmap-output heatmap.html

Metrics: KL divergence + log-odds reduction of correct move.

Heatmap mode (--heatmap): generates per-layer chess.svg boards coloured by
the log-odds effect of patching each square individually. Red dot = 1st PV
move target; green dot = 3rd PV move target. Output as interactive HTML.
"""
import pickle, warnings, argparse, sys, numpy as np, torch, matplotlib.pyplot as plt
import chess, chess.svg
from IPython.display import display, HTML
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, "leela_pytorch_impl")
from model import Lc0Model
from leela_board import LeelaBoard
from utils import sq2idx, idx2sq

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
    p.add_argument("--heatmap", action="store_true",
                   help="Generate per-square heatmap visualisation (skips line-plot)")
    p.add_argument("--heatmap-layer", type=str, default=None,
                   help="Layer name for single-layer heatmap (default: all layers)")
    p.add_argument("--heatmap-puzzles", type=int, nargs="*", default=None,
                   help="Puzzle indices to visualise (0-based), e.g. 0 1 2; "
                        "default: first 3 puzzles with different_targets=True")
    p.add_argument("--heatmap-output", type=str, default=None,
                   help="Path to save heatmap HTML; default: display inline")
    p.add_argument("--heatmap-board-size", type=int, default=300,
                   help="Pixel size of each individual chess board (default: 300)")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Heatmap helpers
# ---------------------------------------------------------------------------

#  SVG helpers — chess.svg always uses a fixed 390×390 viewBox with
#  45×45 px squares and a 15 px margin, regardless of the *size* parameter.
#  The *size* parameter only controls CSS display scaling.

_SVG_MARGIN = 15
_SVG_SQ = 45


def _svg_square_centre(square: int) -> tuple[float, float]:
    """Return (x, y) centre of *square* in chess.svg's fixed 390×390 viewBox."""
    file_idx = chess.square_file(square)   # 0=a … 7=h
    rank_idx = chess.square_rank(square)   # 0=1 … 7=8
    cx = _SVG_MARGIN + (file_idx + 0.5) * _SVG_SQ
    cy = _SVG_MARGIN + (7 - rank_idx + 0.5) * _SVG_SQ   # SVG y=0 is top
    return cx, cy


def _inject_dot(svg_str: str, square: int, colour: str, radius: float = 6) -> str:
    """Add a filled circle on top of *square* inside the SVG markup."""
    cx, cy = _svg_square_centre(square)
    dot = (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" '
        f'fill="{colour}" stroke="#333" stroke-width="1" opacity="0.9"/>'
    )
    return svg_str.replace("</svg>", dot + "\n</svg>")


def _get_principal_variation(model, board: LeelaBoard, n_plies: int = 3
                             ) -> list[str]:
    """Return the model's top-move principal variation (list of UCI strings)."""
    pv = []
    b = board.copy()
    for _ in range(n_plies):
        if b.pc_board.is_game_over():
            break
        policy, _wdl, _ = model.play(b, return_probs=True)
        top = list(model.top_moves(b, policy, top_k=1).keys())
        if not top:
            break
        uci = top[0]
        pv.append(uci)
        b.push_uci(uci)
    return pv


def _build_layer_list(inner, patch_what: str, heatmap_layer: str | None):
    """Return list of (layer_name, module) pairs to iterate over."""
    module_dict = dict(inner.named_modules())
    suffix = {"ffn_skip": "/ffn/skip", "attn_skip": "/mha/out/skip",
              "ln2": "/ln2"}[patch_what]
    names = []
    if f"attn_body{suffix}" in module_dict:
        names.append(f"attn_body{suffix}")
    names += [f"encoder{i}{suffix}" for i in range(N_LAYERS)]

    if heatmap_layer is not None:
        # Filter to the requested layer
        names = [n for n in names if n == heatmap_layer]
        if not names:
            print(f"WARNING: layer '{heatmap_layer}' not found, using all layers")

    pairs = []
    for name in names:
        if name in module_dict:
            pairs.append((name, module_dict[name]))
        else:
            print(f"  WARNING: module '{name}' not found, skipping")
    return pairs


def _colour_map_hex(val: float, vmin: float, vmax: float) -> str:
    """Map a value to a blue–white–red hex colour on a diverging scale.

    Blue  = negative (patching *helped* the correct move).
    White = neutral.
    Red   = positive (patching *hurt* the correct move).
    """
    span = vmax - vmin if vmax != vmin else 1.0
    t = (val - vmin) / span          # 0 at vmin, 1 at vmax
    if vmin < 0 < vmax:
        zero_t = (0 - vmin) / span   # t-value where val == 0
        if t <= zero_t:
            # blue → white: progress from vmin (blue) to 0 (white)
            s = t / zero_t            # 0 at vmin, 1 at zero
            return f"#{int(255*s):02x}{int(255*s):02x}ff"
        else:
            # white → red: progress from 0 (white) to vmax (red)
            s = (t - zero_t) / (1 - zero_t)  # 0 at zero, 1 at vmax
            return f"#ff{int(255*(1-s)):02x}{int(255*(1-s)):02x}"
    elif vmin >= 0:
        # all-positive: white → red
        return f"#ff{int(255*(1-t)):02x}{int(255*(1-t)):02x}"
    else:
        # all-negative: blue → white
        s = t                       # 0 at vmin (blue), 1 at vmax (white)
        return f"#{int(255*s):02x}{int(255*s):02x}ff"


def _build_legend_html(global_vmin: float, global_vmax: float) -> str:
    """Return an HTML/CSS colour bar legend for the blue–white–red scale."""
    # Determine tick positions
    if global_vmin < 0 < global_vmax:
        zero_pct = int(round(-global_vmin / (global_vmax - global_vmin) * 100))
    elif global_vmin >= 0:
        zero_pct = 0
    else:
        zero_pct = 100

    return f"""
    <div style="margin: 12px 0 4px 0; display: flex; align-items: center; gap: 10px;
                font-family: sans-serif; font-size: 0.85em;">
      <span style="color:#888;">Log-odds reduction &mdash;</span>
      <div style="width: 260px; height: 16px; border-radius: 3px; border: 1px solid #aaa;
                  background: linear-gradient(to right, #0000ff 0%, #ffffff {zero_pct}%, #ff0000 100%);">
      </div>
      <span style="color:#3366cc; font-weight:600;">{global_vmin:+.3f}</span>
      <span style="color:#888;">0</span>
      <span style="color:#cc3333; font-weight:600;">{global_vmax:+.3f}</span>
    </div>
    <div style="font-size:0.78em; color:#999; margin-bottom:8px;">
      Blue = patching <i>helped</i> correct move &ensp;|&ensp;
      Red = patching <i>hurt</i> correct move
    </div>"""


def _process_one_puzzle(row, model, inner, module_dict, args, puzzle_num, total):
    """Run per-square patching for one puzzle; return (html_section, board_data)."""
    board_size = args.heatmap_board_size

    # --- Prepare clean & corrupted inputs ---
    cb = LeelaBoard.from_fen(row["FEN"])
    crb = LeelaBoard.from_fen(row["corrupted_fen"])
    ci = model.make_inputs([cb])
    cri = model.make_inputs([crb])

    with torch.no_grad():
        co = inner(ci)
    clean_logits = co[0][0]
    correct_idx = clean_logits.argmax().item()

    # --- Principal variation ---
    pv = _get_principal_variation(model, cb, n_plies=3)
    print(f"  PV: {' '.join(pv)}")
    targets_chess = {}
    turn = cb.pc_board.turn
    if len(pv) >= 1:
        sq_name = idx2sq(sq2idx(pv[0][2:4], turn), turn)
        targets_chess["move1"] = chess.parse_square(sq_name)
        print(f"    1st move target: {sq_name}")
    if len(pv) >= 3:
        sq_name = idx2sq(sq2idx(pv[2][2:4], turn), turn)
        targets_chess["move3"] = chess.parse_square(sq_name)
        print(f"    3rd move target: {sq_name}")

    # --- Build layer list ---
    layers = _build_layer_list(inner, args.patch_what, args.heatmap_layer)
    if not layers:
        return None, None

    # --- Capture corrupted activations (once for all layers) ---
    corrupted_acts = {}
    hooks = []

    def _capture_hook(name):
        def hook(module, inp, out):
            corrupted_acts[name] = out.detach().clone()
        return hook

    for name, _mod in layers:
        hooks.append(module_dict[name].register_forward_hook(_capture_hook(name)))
    with torch.no_grad():
        _ = inner(cri)
    for h in hooks:
        h.remove()

    # --- Per-square patching for each layer ---
    all_squares = list(range(64))
    board_data = []  # (label, svg_str, vmin, vmax)

    for layer_name, mod in tqdm(layers, desc=f"Puzzle {puzzle_num+1}/{total} layers"):
        if layer_name not in corrupted_acts:
            continue
        corr_act = corrupted_acts[layer_name]
        lodd_per_sq = np.full(64, np.nan, dtype=np.float32)

        for sq in all_squares:
            def _patch_one_sq(module, inp, out, *, _sq=sq, _corr=corr_act):
                patched = out.clone()
                patched[_sq] = _corr[_sq]
                return patched

            h = mod.register_forward_hook(_patch_one_sq)
            with torch.no_grad():
                patched_out = inner(ci)
            h.remove()
            lodd_per_sq[sq] = log_odds_reduction(
                clean_logits, patched_out[0][0], correct_idx
            )

        # --- Build colour map (symmetric range around zero) ---
        finite = lodd_per_sq[np.isfinite(lodd_per_sq)]
        if len(finite) == 0:
            vmin, vmax = -1, 1
        else:
            abs_max = max(abs(float(np.min(finite))), abs(float(np.max(finite))))
            abs_max = max(abs_max, 1e-6)
            vmin, vmax = -abs_max, +abs_max

        fills = {}
        for model_sq in all_squares:
            val = lodd_per_sq[model_sq]
            sq_name = idx2sq(model_sq, turn)
            chess_sq = chess.parse_square(sq_name)
            if np.isnan(val) or not np.isfinite(val):
                fills[chess_sq] = "#cccccc"
            else:
                fills[chess_sq] = _colour_map_hex(val, vmin, vmax)

        # --- Render SVG ---
        svg = chess.svg.board(cb.pc_board, fill=fills, size=board_size)
        if "move1" in targets_chess:
            svg = _inject_dot(svg, targets_chess["move1"], "red", radius=7)
        if "move3" in targets_chess:
            svg = _inject_dot(svg, targets_chess["move3"], "green", radius=7)

        if "attn_body" in layer_name:
            label = "embed+FFN"
        elif "encoder" in layer_name:
            label = layer_name.split("encoder")[1].split("/")[0]
            label = f"enc {label}"
        else:
            label = layer_name

        board_data.append((label, svg, vmin, vmax))

    # --- Corrupted board (reference, no heatmap) ---
    corr_svg = chess.svg.board(crb.pc_board, size=board_size)
    if "move1" in targets_chess:
        corr_svg = _inject_dot(corr_svg, targets_chess["move1"], "red", radius=6)
    if "move3" in targets_chess:
        corr_svg = _inject_dot(corr_svg, targets_chess["move3"], "green", radius=6)
    board_data.insert(0, ("corrupted", corr_svg, None, None))

    # --- Build HTML section for this puzzle ---
    all_vmins = [vmin for _, _, vmin, _ in board_data if vmin is not None]
    all_vmaxs = [vmax for _, _, _, vmax in board_data if vmax is not None]
    global_vmin = min(all_vmins) if all_vmins else 0
    global_vmax = max(all_vmaxs) if all_vmaxs else 1
    legend_html = _build_legend_html(global_vmin, global_vmax)

    fen_display = row["FEN"]
    if len(fen_display) > 85:
        fen_display = fen_display[:82] + "..."

    puzzle_html = [
        f"<h3>Puzzle {puzzle_num+1}/{total}</h3>",
        f"<p><b>FEN:</b> {fen_display} &nbsp;|&nbsp; "
        f"<b>PV:</b> {' '.join(pv)} &nbsp;|&nbsp; "
        f"<span style='color:red;'>● 1st target</span> &nbsp; "
        f"<span style='color:green;'>● 3rd target</span></p>",
        legend_html,
        "<div style='display: flex; flex-wrap: wrap; gap: 20px;'>",
    ]
    for label, svg, vmin, vmax in board_data:
        if vmin is None:
            puzzle_html.append(
                f"<div style='text-align:center; border:1px solid #ddd; "
                f"border-radius:8px; padding:8px; background:#fafafa;'>"
                f"<b>{label}</b><br>"
                f"<span style='font-size:0.8em; color:#888;'>"
                f"corrupted position</span><br>"
                f"{svg}</div>"
            )
        else:
            puzzle_html.append(
                f"<div style='text-align:center; border:1px solid #ddd; "
                f"border-radius:8px; padding:8px; background:#fafafa;'>"
                f"<b>{label}</b><br>"
                f"<span style='font-size:0.8em; color:#888;'>"
                f"range [{vmin:+.3f}, {vmax:+.3f}]</span><br>"
                f"{svg}</div>"
            )
    puzzle_html.append("</div>")

    return "\n".join(puzzle_html), board_data


def heatmap_main(args):
    """Run per-square activation patching and render chess.svg heatmaps
    for one or more puzzles."""
    print("Loading puzzles...")
    with open("leela_interp/interesting_puzzles.pkl", "rb") as f:
        puzzles = pickle.load(f)

    puzzles = puzzles[puzzles["corrupted_fen"].notna()].copy()
    print(f"Total with corrupted FEN: {len(puzzles)}")

    puzzles_diff = puzzles[puzzles["different_targets"] == True]
    if len(puzzles_diff) == 0:
        puzzles_diff = puzzles

    # Determine which puzzle indices to process
    if args.heatmap_puzzles:
        indices = [min(i, len(puzzles_diff) - 1) for i in args.heatmap_puzzles]
    else:
        n = min(3, len(puzzles_diff))
        indices = list(range(n))

    total = len(indices)
    print(f"Processing {total} puzzle(s): indices {indices}")

    print(f"Loading model: {args.model}")
    model = Lc0Model(args.model, device=args.device)
    inner = model._lc0_model
    module_dict = dict(inner.named_modules())

    # Pre-validate layers so we only print this once
    layers = _build_layer_list(inner, args.patch_what, args.heatmap_layer)
    if not layers:
        print("No valid layers found, exiting.")
        return
    print(f"Rendering heatmaps for {len(layers)} layer(s) "
          f"(patch_what={args.patch_what})")

    puzzle_sections = []
    for i, idx in enumerate(indices):
        row = puzzles_diff.iloc[idx]
        print(f"\n--- Puzzle {i+1}/{total} (index {idx}) ---")
        print(f"  Clean FEN: {row['FEN']}")
        section, _ = _process_one_puzzle(row, model, inner, module_dict,
                                          args, puzzle_num=i, total=total)
        if section:
            puzzle_sections.append(section)

    # --- Assemble final HTML ---
    model_name = args.model.split("/")[-1]
    html_parts = [
        "<div style='font-family: sans-serif; max-width: 1400px;'>",
        f"<h2>Activation‑Patching Heatmap</h2>",
        f"<p><b>Model:</b> {model_name} &nbsp;|&nbsp; "
        f"<b>Patch:</b> {args.patch_what} &nbsp;|&nbsp; "
        f"<b>Puzzles:</b> {total}</p>",
    ]
    for section in puzzle_sections:
        html_parts.append(f"<hr style='margin: 20px 0;'>")
        html_parts.append(section)
    html_parts.append("</div>")
    full_html = "\n".join(html_parts)

    out_path = args.heatmap_output
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        print(f"\nHeatmap HTML saved to {out_path}")
    else:
        display(HTML(full_html))

    print("Done.")


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.heatmap:
        heatmap_main(args)
        return

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
