# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: chess-interp (3.14.4)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: Leela Interp Demo with NNsight
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .qmd
#       format_name: quarto
#       format_version: '1.0'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: chess-interp (3.14.4)
#     language: python
#     name: python3
# ---
#
#
#
# ## Setup

# %%
import sys
import os
from pathlib import Path

# Import modules from parent and sibling directories
current_dir = Path.cwd().resolve()
repo_root = current_dir.parent if current_dir.name == "leela_interp" else current_dir
leela_pytorch_impl_dir = repo_root / "leela_pytorch_impl"

for path in (repo_root, leela_pytorch_impl_dir):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.append(path_str)

from model import Lc0Model
from leela_board import LeelaBoard
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

# %%
ONNX_MODEL_NAME = 'lc0.onnx'
lc0 = Lc0Model(onnx_model_path=os.path.join(leela_pytorch_impl_dir, ONNX_MODEL_NAME))

# %% [markdown]
# ### Hugging Face Authentication
#
# We load local `.env` variables and initialize our token to fetch the chess evaluation sets from Hugging Face properly.

# %%
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()  # Load environment variables from .env file

login(token=os.getenv("HUGGINGFACE_TOKEN"))

# %% [markdown]
# ## Dataset
#
# We will use a sample chess dataset to run our first probe on the model

# %%
from datasets import load_dataset

print("1. Load actual dataset (streaming)")
try:
    # Trying lowercase to see if it fixes the typo
    data = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
except Exception as e:
    print(f"Failed to load: {e}")

# %% [markdown]
# ### Streaming the Dataset & Filtering
#
# We stream the dataset from Hugging Face instead of downloading it fully into memory to prevent crashing. We also selectively filter the incoming stream to extract exactly 500 puzzle positions.

# %%
import chess
def get_nth(iterable, n, default=None):
    for i, x in enumerate(iterable):
        if i == n:
            return x
    return default

item = get_nth(data, 3)
fen = item['FEN']
board = LeelaBoard.from_fen(fen)

# %%
item

# %%
board


# %%
def get_nth_move(board: LeelaBoard, n:int):
    lookahead = board.copy()
    # Policy, Win-Loss-Draw, Moves Left Ahead
    for i in range(n) :
        policy, wld, mlh = lc0.play(lookahead)
        # top_moves returns a dictionary of <moves, probabilities>, we only want the keys
        top_moves = list(lc0.top_moves(lookahead, policy).keys())
        # presumably this means there are no more moves to make
        if len(top_moves) == 0:
            return None
             
        best_move = top_moves[0]
        lookahead.push_uci(best_move)
    
    return lookahead


# %%
copy = board.copy()
copy.push_uci(item['Moves'].split(' ')[0])
# copy

# %%
policy, wld , mlh  = lc0.play(board)

# %%
print("2. Extract Boards and Labels (Balanced)")
boards = []
num_boards = 0

target_num = 1000

baseline = LeelaBoard()

# These arrays will contain a number 0-63 that represent a square on the chessboard
first_source_indexes = np.zeros(target_num, dtype=int)
first_target_indexes = np.zeros(target_num, dtype=int)
lookahead_source_indexes = np.zeros(target_num, dtype=int)
lookahead_target_indexes = np.zeros(target_num, dtype=int)

# must be minimum 2
LOOKAHEAD_NUM = 1+1

known_fens = set()
for i, item in enumerate(data):
    fen = item.get('FEN', item.get('fen'))
    if fen in known_fens:
        continue;
    else:
        known_fens.add(fen)
        
    board = LeelaBoard.from_fen(fen)
    # Ex: ["e2e4", "e7e5"]
    moves = item['Moves'].split(' ')

    # 1 because the puzzle must start with a move
    # *2 because a "move" is a pair of moves
    if len(moves)<1+LOOKAHEAD_NUM*2:
        continue;

    # Puzzle data starts one move from the start of the puzzl
    # This move is given to the player. i.e. it is played by enemy
    start_uci = moves[0]
    board.push_uci(start_uci)
    boards.append(board)
    # We consider this the first real move.
    # This is the first move that a player would have to find
    first_uci = moves[1]
    first_source_indexes[num_boards] = baseline.sq2idx(first_uci[0:2])
    first_target_indexes[num_boards] = baseline.sq2idx(first_uci[2:4])

    # in the case where the game should end before lookahead don't include
    try:
        lookahead_uci = moves[2*LOOKAHEAD_NUM]
    except Exception as e:
        print(moves[2])
        continue
    
    lookahead_source_indexes[num_boards] = baseline.sq2idx(lookahead_uci[0:2])
    lookahead_target_indexes[num_boards] = baseline.sq2idx(lookahead_uci[2:4])

    num_boards += 1

    if num_boards >= target_num:
        break;

print(f"Found {num_boards} positions.")

# %%
labels = np.array(lookahead_target_indexes, dtype=int)

# %%
print("3. Force model and inputs to CPU")
lc0.cpu() # ensure everything is on CPU to avoid device mismatch
lc0._device = 'cpu'
inputs = lc0.make_inputs(boards).cpu()

# %% [markdown]
# ## Extracting Activations
#
# Here we construct inputs for the model and run the PyTorch forward pass. We use NNsight to intercept the outputs of the Multi-Layer Perceptron (MLP) at our designated probe layer (`LAYER_TO_PROBE = 10`). 

# %%
from nnsight import NNsight

print("Run forward pass and extract activations with NNsight...")
# has 15 layers total
LAYER_TO_PROBE = 10

# Wrap our model in NNsight
nnsight_model = NNsight(lc0)

with nnsight_model.trace(inputs) as tracer:
    # Grab the output of the residual stream
    hidden_states = nnsight_model._lc0_model.post_mlp[LAYER_TO_PROBE].output.save()
    

print("Extracting features")
X_activations = hidden_states if isinstance(hidden_states, torch.Tensor) else hidden_states.value
print(f"Activations shape: {X_activations.shape}")

batch_idx = torch.arange(target_num, dtype=int)
i1 = torch.from_numpy(first_source_indexes)
i2 = torch.from_numpy(first_target_indexes)

# Represent the hiddens layers for the targets
# Should be of the form [batches, h_dim]
first_source_h = X_activations[batch_idx, i1].to("cpu")
first_target_h = X_activations[batch_idx, i2].to("cpu")

# %% [markdown]
# # Finding the shape
#
# What is the shape the Query and Key Matricies?

# %%
for name, module in lc0.named_modules():
    print(name)
    print(module)

# %%
with nnsight_model.trace(inputs) as tracer:
    # Grab the output of the residual stream
    step = getattr(nnsight_model._lc0_model, 'encoder10/mha/Q/w')
    hidden_states = step.output.save()

print(hidden_states)
print(f"shape: {hidden_states.shape}")

# %% [markdown]
# shape is of the form [BATCH_SIZE * 64, H_DIM]
#
# In other words, the matricies are really (64, 768)
#
# # Training the Linear Probe
#
# After capturing the activations from our hidden layer, we split the data and train a Logistic Regression classifier (`Linear Probe`) with 5-fold cross-validation on the intermediate layer's geometry to see if it implicitly "understands" the concept of 'check'.

# %%
import torch
import torch.nn as nn

class BilinearProbe(nn.Module):
    def __init__(self, d_attn, d_h):
        """
        d_attn: 64  — attention/bottleneck dimension (also num squares)
        d_h:   768  — hidden state dimension
        """
        super().__init__()
        self.U = nn.Parameter(torch.randn(d_attn, d_h) * 0.01)  # (64, 768)
        self.V = nn.Parameter(torch.randn(d_attn, d_h) * 0.01)  # (64, 768)
        self.c = nn.Parameter(torch.zeros(d_attn))              # (64,)

    def forward(self, h_y_all, h_t1):
        """
        h_y_all: (d_attn, d_h)  — h^L for each candidate square y
        h_t1:    (d_h,)     — h^L at position t1
        Returns:  (d_attn,)     — one logit per square, ready to softmax
        """
        # U h_y^T -> (64, 64): project each candidate square
        Uh = h_y_all @ self.U.T      # (64, 768) @ (768, 64) -> (64, 64)

        # V h_t1 -> (64,): project the conditioning token
        Vh = self.V @ h_t1           # (64, 768) @ (768,) -> (64,)

        # bilinear scores: each row of Uh dotted with Vh
        logits = Uh @ Vh + self.c    # (64, 64) @ (64,) -> (64,) + (64,)

        return logits
    
    def predict(self, h_y_all, h_t1):
        logits = self.forward(h_y_all, h_t1)
        return torch.softmax(logits, dim=0)


# %%
probe = BilinearProbe(d_attn=64, d_h=768)
optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()  # expects raw logits, applies softmax internally

def train_step(h_y_all, h_t1_batch, t3_labels):
    """
    h_y_all:     (64, 768)  — fixed, the h^L for each square
    h_t1_batch:  (batch, 768)  — h^L at t1 for each game in the batch
    t3_labels:   (batch,)   — ground truth target square index (0-63)
    """
    probe.train()
    optimizer.zero_grad()

    # run probe for each item in batch
    logits = torch.stack([
        probe(h_y_all, h_t1) for h_t1 in h_t1_batch
    ])  # (batch, 64)

    loss = loss_fn(logits, t3_labels)
    loss.backward()
    optimizer.step()
    return loss.item()

labels_torch = torch.from_numpy(labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("gpu available" if torch.cuda.is_available() else "No gpu")
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

first_target_h = first_target_h.to(device)
labels_torch = labels_torch.to(device)
X_activations = X_activations.to(device)
probe = probe.to(device)

# training loop
num_epochs = 100
for epoch in range(num_epochs):
    for i in range(num_boards//5):
        loss = train_step(X_activations[i], first_target_h, labels_torch)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

# %%
probe.eval()
correct = 0
total = 0

with torch.no_grad():
    for i in range(num_boards//5, num_boards):
        probs = probe.predict(X_activations[i], first_target_h[i])  # (64,)
        predicted = probs.argmax()
        correct += (predicted == labels_torch[i]).item()
        total += 1

print(f"Accuracy: {correct/total:.4f} ({correct}/{total})")

# %% [markdown]
# ## Visualization
#
# Finally, we visualize the probe's predictions on a sample of the unseen test set natively in the notebook. This compares the actual board position directly against the prediction class from our Logistic Regression logic. Red text indicates an error, while green indicates correct classification.

# %%
from IPython.display import display, HTML
import chess.svg

print("Visualizing probe predictions on TEST SET...")

try:
    # Get the probability that the position is in check (class 1) from the test set
    probs = probe.predict_proba(X_test)[:, 1]

    # Find some indices to plot (within the test set arrays)
    test_in_check_idx = np.where(y_test == 1)[0][:5]
    test_not_in_check_idx = np.where(y_test == 0)[0][:5]

    html_out = "<div style='display: flex; flex-direction: column; gap: 20px;'>"

    def render_group(test_indices, title):
        html = f"<h3>{title}</h3><div style='display: flex; flex-wrap: wrap; gap: 15px;'>"
        for t_idx in test_indices:
            # Map back to the original index in `boards`
            orig_idx = idx_test[t_idx] 
            
            board = boards[orig_idx].pc_board
            prob = probs[t_idx]
            
            # Color the probability green if correct, red if incorrect
            is_correct = (prob > 0.5 and y_test[t_idx] == 1) or (prob <= 0.5 and y_test[t_idx] == 0)
            color = "green" if is_correct else "red"
            
            svg = chess.svg.board(board, size=200)
            html += f"""
            <div style="border: 1px solid #ccc; padding: 10px; border-radius: 8px; text-align: center; background: white; color: black;">
                {svg}
                <div style="margin-top: 10px; font-family: sans-serif;">
                    <b>Prob (has_passed_pawn):</b> <span style="color: {color}">{prob:.1%}</span><br/>
                    <b>True Label:</b> {bool(y_test[t_idx])}
                </div>
            </div>
            """
        html += "</div>"
        return html

    html_out += render_group(test_in_check_idx, "5 Test Positions IN CLASS")
    html_out += render_group(test_not_in_check_idx, "5 Test Positions NOT IN CLASS")
    html_out += "</div>"

    display(HTML(html_out))

except NameError as e:
    print(f"Error: Make sure you ran the previous training cell first. ({e})")

# %% [markdown]
# ## Visualizing Attention Outputs
#
# TODO: VISUALIZE ATTENTION FOR THE PAWNS THAT ARE PASSED
#
# Here we extract the inner attention matrices from Leela's ONNX-converted Transformer layer using `nnsight`. Since King safety is directly correlated to the concept of check, we map how much the `King` query token attends to other pieces (keys) on the board!

# %%
from IPython.display import display, HTML

# Define available layers and heads in the model
AVAILABLE_ATTENTION = {
    "layers": [f"encoder{i}" for i in range(10)], # encoder0 up to encoder9
    "heads": list(range(24)),
    "description": "24 heads per layer across 10 encoder layers."
}

def plot_attention_for_square(board_idx, query_square, layer_name="encoder7", head_idx=None):
    """
    Extracts and visualizes the attention weights for a given square on the board.
    If head_idx is None, averages across all 24 heads.
    """
    board_demo = boards[board_idx]
    
    # Forward pass on ONE board
    single_input = inputs[board_idx:board_idx+1]
    
    with nnsight_model.trace(single_input) as tracer:
        # Construct module path dynamically
        module_path = f"{layer_name}/mha/QK/softmax"
        layer = getattr(nnsight_model._lc0_model, module_path)
        attn_out = layer.output.save()
        
    weights = attn_out if isinstance(attn_out, torch.Tensor) else attn_out.value 
    weights = weights[0].cpu().numpy() # [24, 64, 64]
    
    if head_idx is None:
        mean_attn = weights.mean(axis=0)[query_square] # average across all 24 heads
    else:
        mean_attn = weights[head_idx][query_square] # specific head

    # Normalize between 0 and 1 so it fits a color scale smoothly
    min_val, max_val = mean_attn.min(), mean_attn.max()
    norm_attn = (mean_attn - min_val) / (max_val - min_val + 1e-9)

    colors = {}
    for sq in range(64):
        val = float(norm_attn[sq])
        # White to Red heatmap background
        color_hex = f"#{int(255):02x}{int(255*(1-val)):02x}{int(255*(1-val)):02x}"
        colors[sq] = color_hex

    svg = chess.svg.board(
        board_demo.pc_board,
        fill=colors,
        size=400
    )

    # Put a nice title above
    head_title = "Average across all 24 heads" if head_idx is None else f"Head {head_idx}"
    html_out = f"<h3>Attention Map: {layer_name}, {head_title} | Query square: {chess.square_name(query_square).upper()}</h3>"
    html_out += svg
    display(HTML(html_out))

print(f"Available layers to choose from: {AVAILABLE_ATTENTION['layers']}")
print(f"Available heads to choose from: {AVAILABLE_ATTENTION['heads'][0]} to {AVAILABLE_ATTENTION['heads'][-1]}")

# %%
demo_board_indices = [idx_test[t_idx] for t_idx in test_in_check_idx]
demo_idx = next(
    (board_idx for board_idx in demo_board_indices if boards[board_idx].pc_board.turn == chess.WHITE),
    demo_board_indices[0] # fallback
)

print("Visualizing White King query:")
w_king_sq = boards[demo_idx].pc_board.king(chess.WHITE)
plot_attention_for_square(demo_idx, w_king_sq, layer_name="encoder7", head_idx=None)

print("\nVisualizing Black King query (on the same board):")
b_king_sq = boards[demo_idx].pc_board.king(chess.BLACK)
plot_attention_for_square(demo_idx, b_king_sq, layer_name="encoder7", head_idx=None)
