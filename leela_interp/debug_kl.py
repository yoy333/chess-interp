"""Quick diagnostic: baseline KL between clean and corrupted policies."""
import warnings; warnings.filterwarnings('ignore')
import pickle, sys, torch
sys.path.insert(0, 'leela_pytorch_impl')
from model import Lc0Model
from leela_board import LeelaBoard

with open('leela_interp/interesting_puzzles.pkl', 'rb') as f:
    puzzles = pickle.load(f)

p = puzzles[puzzles['corrupted_fen'].notna() & puzzles['different_targets']]
print(f'Puzzles with different_targets: {len(p)}')

for model_path in ['leela_pytorch_impl/lc0-original.onnx', 'leela_pytorch_impl/BT4-tf13tune.onnx']:
    print(f'\n=== {model_path} ===')
    model = Lc0Model(model_path, device='cpu')
    kls = []
    for _, row in p.head(10).iterrows():
        cb = LeelaBoard.from_fen(row['FEN'])
        crb = LeelaBoard.from_fen(row['corrupted_fen'])
        ci = model.make_inputs([cb])
        cri = model.make_inputs([crb])
        with torch.no_grad():
            co = model._lc0_model(ci)
            cro = model._lc0_model(cri)
        cp = torch.softmax(co[0][0], -1)
        crp = torch.softmax(cro[0][0], -1)
        kl = (cp.clamp(1e-9)*(cp.clamp(1e-9).log()-crp.clamp(1e-9).log())).sum().item()
        kls.append(kl)
        print(f'  {row["PuzzleId"]}: KL(clean||corrupt)={kl:.4f}')
    print(f'  Mean KL: {torch.tensor(kls).mean():.4f}')

