import glob
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from music_common import VOCAB, MelodyLSTM, midi_to_tokens

BASE = os.path.dirname(os.path.abspath(__file__))
MIDI_DIR = os.path.join(BASE, "data", "midi")
MODEL_PATH = os.path.join(BASE, "models", "melody_lstm.pt")

SEQ_LEN = 128          # 128 steps = 8 bars of 4/4
STRIDE = 64
BATCH = 128
EPOCHS = 25
MAX_MINUTES = 20       # training stops by itself after this long
PATIENCE = 4           # stop early if no improvement for this many epochs

CONFIG = {"vocab": VOCAB, "embed": 64, "hidden": 256, "layers": 2, "dropout": 0.3}


def make_windows(items):
    xs, ys = [], []
    for seq, mode_tok in items:
        if len(seq) < 32:
            continue
        if len(seq) < SEQ_LEN + 1:
            seq = np.tile(seq, int(np.ceil((SEQ_LEN + 1) / len(seq))))
        for start in range(0, len(seq) - SEQ_LEN + 1, STRIDE):
            chunk = seq[start:start + SEQ_LEN]
            full = np.concatenate(([mode_tok], chunk))   # mood token first
            xs.append(full[:-1])
            ys.append(full[1:])
    return torch.tensor(np.array(xs)), torch.tensor(np.array(ys))


def run_epoch(model, loader, loss_fn, opt=None):
    training = opt is not None
    model.train(training)
    total, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, VOCAB), y.reshape(-1))
            if training:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += loss.item() * y.numel()
            n += y.numel()
            correct += (logits.argmax(-1) == y).sum().item()
    return total / n, correct / n


def main():
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(os.cpu_count() or 4)

    files = glob.glob(os.path.join(MIDI_DIR, "*.mid")) + glob.glob(os.path.join(MIDI_DIR, "*.midi"))
    if not files:
        print("No MIDI files found in", MIDI_DIR)
        return
    print(f"Reading {len(files)} MIDI files...")
    items = []
    for f in files:
        r = midi_to_tokens(f)
        if r is not None:
            items.append(r)
    print(f"Usable tunes: {len(items)} of {len(files)}")
    if len(items) < 20:
        print("Too few usable tunes. Check the dataset.")
        return

    random.shuffle(items)
    cut = int(len(items) * 0.9)
    x_tr, y_tr = make_windows(items[:cut])
    x_va, y_va = make_windows(items[cut:])
    print(f"Training windows: {len(x_tr)}, validation windows: {len(x_va)}")

    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_va, y_va), batch_size=BATCH)

    model = MelodyLSTM(**CONFIG)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    best, bad_epochs, t0 = float("inf"), 0, time.time()
    print("Training started. Press Ctrl+C any time; the best model so far is already saved.\n")
    try:
        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, opt)
            va_loss, va_acc = run_epoch(model, val_loader, loss_fn)
            sched.step(va_loss)
            mins = (time.time() - t0) / 60
            print(f"Epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.2%} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.2%} | {mins:.1f} min", flush=True)

            if va_loss < best:
                best, bad_epochs = va_loss, 0
                torch.save({"state_dict": model.state_dict(), "config": CONFIG}, MODEL_PATH)
                print("   saved best model")
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    print("No improvement, stopping early.")
                    break
            if mins >= MAX_MINUTES:
                print("Time limit reached, stopping.")
                break
    except KeyboardInterrupt:
        print("\nStopped by you.")

    if os.path.exists(MODEL_PATH):
        print("\nDone! Model saved at", MODEL_PATH)
    else:
        print("\nNo model was saved yet. Let at least one epoch finish.")


if __name__ == "__main__":
    main()