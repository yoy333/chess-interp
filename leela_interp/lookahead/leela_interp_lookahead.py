from numpy import float16

def printonce(thing):
    from os import getenv
    rank = int(getenv("LOCAL_RANK"))
    if rank == 0:
        print(thing)

def generateActivations():
    import sys
    import os
    from pathlib import Path

    current_dir = Path.cwd().resolve()
    repo_root = current_dir.parent.parent if current_dir.name == "lookahead" else current_dir
    leela_pytorch_impl_dir = repo_root / "leela_pytorch_impl"

    for path in (repo_root, leela_pytorch_impl_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.append(path_str)

    from model import Lc0Model
    from leela_board import LeelaBoard
    import torch
    import numpy as np

    # %%
    ONNX_MODEL_NAME = 'lc0.onnx'
    lc0 = Lc0Model(onnx_model_path=os.path.join(leela_pytorch_impl_dir, ONNX_MODEL_NAME))

    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()  # Load environment variables from .env file

    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    from datasets import IterableDataset, iterable_dataset, load_dataset

    printonce("1. Load actual dataset (streaming)")
    try:
        data = load_dataset("Lichess/chess-puzzles", split="train", streaming=True)
    except Exception as e:
        printonce(f"Failed to load: {e}")

    printonce("2. Extract Boards and Labels (Balanced)")
    boards = []
    num_boards = 0

    target_num = 100

    # These arrays will contain a number 0-63 that represent a square on the chessboard
    first_source_indexes = np.zeros(target_num, dtype=int)
    first_target_indexes = np.zeros(target_num, dtype=int)
    lookahead_source_indexes = np.zeros(target_num, dtype=int)
    lookahead_target_indexes = np.zeros(target_num, dtype=int)

    # must be minimum 2
    LOOKAHEAD_NUM = 2

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
        first_source_indexes[num_boards] = board.sq2idx(first_uci[0:2])
        first_target_indexes[num_boards] = board.sq2idx(first_uci[2:4])

        # in the case where the game should end before lookahead don't include
        try:
            lookahead_uci = moves[2*LOOKAHEAD_NUM]
        except Exception as e:
            printonce(moves[2])
            continue
        
        lookahead_source_indexes[num_boards] = board.sq2idx(lookahead_uci[0:2])
        lookahead_target_indexes[num_boards] = board.sq2idx(lookahead_uci[2:4])

        num_boards += 1

        if num_boards >= target_num:
            break;

    printonce(f"Found {num_boards} positions.")

    # %%
    labels = np.array(lookahead_target_indexes, dtype=int)

    # %%
    printonce("3. Force model and inputs to CPU")
    lc0.cpu() # ensure everything is on CPU to avoid device mismatch
    lc0._device = 'cpu'
    inputs = lc0.make_inputs(boards).cpu()

    # %%
    from nnsight import NNsight

    printonce("Run forward pass and extract activations with NNsight...")
    # has 15 layers total (0-14)
    LAYER_TO_PROBE = 10

    # Wrap our model in NNsight
    nnsight_model = NNsight(lc0)

    with nnsight_model.trace(inputs) as tracer:
        # Grab the output of the residual stream
        hidden_states = nnsight_model._lc0_model.post_mlp[LAYER_TO_PROBE].output.save()

    printonce("Extracting features")
    X_activations = hidden_states if isinstance(hidden_states, torch.Tensor) else hidden_states.value
    printonce(f"Activations shape: {X_activations.shape}")

    batch_idx = torch.arange(target_num, dtype=int)
    i1 = torch.from_numpy(first_source_indexes)
    i2 = torch.from_numpy(first_target_indexes)

    # Represent the hiddens layers for the targets
    # Should be of the form [batches, h_dim]
    first_source_h = X_activations[batch_idx, i1].to("cpu")
    first_target_h = X_activations[batch_idx, i2].to("cpu")

    # TODO: fix to be correct type in first place

    X_activations = X_activations.to(torch.float16)
    first_target_h = first_target_h.to(torch.float16)
    labels_torch = torch.from_numpy(labels)
    

    return X_activations, first_target_h, labels_torch

def training_loop(X_activations, first_target_h, labels_torch):
    # ---- Data ----
    # X_activations: (N, num_squares, d_hidden)
    # first_target_h: (N, d_hidden)
    # labels_torch: (N,) — long tensor of correct square index
    import torch
    import torch.nn as nn

    class BilinearProbe(nn.Module):
        def __init__(self, d_attn, d_h, num_squares = 64):
            """
            d_attn: 64  — attention/bottleneck dimension (also num squares)
            d_h:   768  — hidden state dimension
            """
            super().__init__()
            self.U = nn.Parameter(torch.randn(d_attn, d_h) * 0.01)  # (64, 768)
            self.V = nn.Parameter(torch.randn(d_attn, d_h) * 0.01)  # (64, 768)
            self.c = nn.Parameter(torch.zeros(num_squares))              # (64,)

        def forward(self, h_y_all, h_t1):
            """
            h_y_all: (B, num_sqaures, d_h)  — h^L for each candidate square y
            h_t1:    (B, d_h)     — h^L at position t1
            Returns:  (B, num_squares)     — one logit per square, ready to softmax
            """
            # Project candidate squares: (B, num_squares, d_h) @ (d_h, d_attn) -> (B, num_squares, d_attn)
            Uh = h_y_all @ self.U.T

            # Project conditioning token: (B, d_h) @ (d_h, d_attn) -> (B, d_attn)
            Vh = h_t1 @ self.V.T

            # (B, num_sqaures, d_attn) @ (B, d_attn, 1) -> (B, num_squares, 1) -> (B, num_sqaures)
            logits = (Uh @ Vh.unsqueeze(-1))\
            .squeeze(-1) + self.c
            return logits
        
        def predict(self, h_y_all, h_t1):
            """
            Same input shapes as forward(). Returns (B, d_attn) softmax probabilities
            (or (d_attn,) if the inputs were unbatched).
            """
            logits = self.forward(h_y_all, h_t1)
            probs = torch.softmax(logits, dim=-1)
            return probs


    # %%
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    # ---- Config ----
    d_attn = 32
    d_h = 768
    num_squares = 64
    batch_size = 32
    num_epochs = 5
    lr = 1e-2
    weight_decay = 0
    val_split = 0.3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dataset = TensorDataset(X_activations, first_target_h, labels_torch)

    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # ---- Model / Optimizer / Loss ----
    probe = BilinearProbe(d_attn=d_attn, d_h=d_h, num_squares=num_squares)#.to(device)
    optimizer_strat = optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()  # expects raw logits + integer class labels

    # deepspeed
    import deepspeed
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "ds_config.json")

    model_engine, optimizer, data_loader, _ = deepspeed.initialize(
        model=probe,
        model_parameters=probe.parameters(),
        config=config_path,
        training_data = train_set,
    )

    def evaluate(model_engine, loader):
        model_engine.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for h_y_all, h_t1, labels in loader:
                d = model_engine.device
                h_y_all, h_t1, labels = h_y_all.to(d), h_t1.to(d), labels.to(d)

                logits = model_engine(h_y_all, h_t1)
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    for epoch in range(num_epochs):
        model_engine.train()
        running_loss, correct, total = 0.0, 0, 0

        for h_y_all, h_t1, labels in data_loader:
            d = model_engine.device
            h_y_all = h_y_all.to(d)      # (B, num_squares, d_h)
            h_t1 = h_t1.to(d)            # (B, d_h)
            labels = labels.to(d)        # (B,)

            logits = model_engine(h_y_all, h_t1)     # (B, num_squares)
            loss = criterion(logits, labels)
            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model_engine, val_loader)

        printonce(f"Epoch {epoch+1:02d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
    
def main():
    activations = generateActivations()
    training_loop(*activations)
    

if __name__ == "__main__":
    main()
