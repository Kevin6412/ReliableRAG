"""Small trainable evidence sufficiency MLP with validation-only calibration."""
from __future__ import annotations
import random
import numpy as np


def _torch():
    import torch
    from torch import nn
    return torch, nn


def make_model(input_dim: int, hidden: int = 32):
    torch, nn = _torch()
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))


def fit(train_x, train_y, val_x, val_y, *, epochs=40, lr=1e-3, seed=42):
    torch, nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    mean = train_x.mean(axis=0)
    std = np.maximum(train_x.std(axis=0), 1e-6)
    xtr = torch.from_numpy(((train_x-mean)/std).astype(np.float32))
    xval = torch.from_numpy(((val_x-mean)/std).astype(np.float32))
    ytr = torch.from_numpy(train_y.astype(np.float32)).reshape(-1, 1)
    yval = torch.from_numpy(val_y.astype(np.float32)).reshape(-1, 1)
    model = make_model(train_x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    weight = float((ytr.numel()-ytr.sum().item())/max(1., ytr.sum().item()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
    best_loss = float('inf')
    best_state = None
    patience_left = 8
    for epoch in range(epochs):
        model.train()
        indices = torch.randperm(len(xtr))
        for batch in indices.split(64):
            optimizer.zero_grad()
            loss = loss_fn(model(xtr[batch]), ytr[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xval), yval).item()
        print(f'epoch {epoch+1:02d}  validation_loss={val_loss:.5f}')
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k,v in model.state_dict().items()}
            patience_left = 8
        else:
            patience_left -= 1
            if patience_left == 0:
                break
    model.load_state_dict(best_state)
    return model, mean, std


def probabilities(model, xs, mean, std):
    torch, _ = _torch()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor((xs-mean)/std, dtype=torch.float32))
        return torch.sigmoid(logits).squeeze(-1).numpy()


def balanced_accuracy(y, pred) -> float:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    recalls = [float(np.mean(pred[y] == True)) if np.any(y) else None,
               float(np.mean(pred[~y] == False)) if np.any(~y) else None]
    vals = [v for v in recalls if v is not None]
    return float(np.mean(vals)) if vals else 0.


def best_threshold(y, prob, grid=None):
    if grid is None:
        grid = np.linspace(.05, .95, 37)
    scores = [(balanced_accuracy(y, prob >= th), -abs(th-0.5), float(th)) for th in grid]
    return max(scores)[-1]


def save(path, model, mean, std, *, model_name, k, noise_rates,
         neural_threshold, similarity_threshold, filter_threshold):
    torch, _ = _torch()
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'mean': mean.tolist(),
                'std': std.tolist(), 'input_dim': len(mean),
                'encoder_name': model_name, 'k': k, 'noise_rates': noise_rates,
                'neural_threshold': float(neural_threshold),
                'similarity_threshold': float(similarity_threshold),
                'filter_threshold': float(filter_threshold)}, path)


def load(path):
    torch, _ = _torch()
    # weights_only=True prevents loading arbitrary pickled objects.
    obj = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model(obj['input_dim'])
    model.load_state_dict(obj['state_dict'])
    model.eval()
    return model, np.asarray(obj['mean'], dtype=np.float32), np.asarray(obj['std'], dtype=np.float32), obj
