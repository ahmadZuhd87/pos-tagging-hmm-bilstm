"""
train.py — Training / evaluation loop for the BiLSTM tagger.

Hand-written training loop (no Trainer abstraction) so every engineering choice
is visible and explainable:
    * explicit seeding for reproducibility (spec requires reproducible results)
    * masked cross-entropy loss (ignore PAD positions)
    * gradient clipping
    * early stopping on dev macro-F1 (not accuracy — we optimise the metric the
      spec cares about for rare tags)
    * ReduceLROnPlateau scheduling
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import copy
import torch
import torch.nn as nn

from data import Sentence, Vocab, make_batches, PAD_IDX
from metrics import compute_metrics, metrics_summary


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5     # L2 regularisation
    grad_clip: float = 5.0
    patience: int = 5              # early stopping patience (epochs)
    seed: int = 42


@torch.no_grad()
def evaluate(model, sentences: List[Sentence], vocab: Vocab,
             device, batch_size: int = 64):
    model.eval()
    y_true, y_pred = [], []
    for word_ids, char_ids, tag_ids, mask in make_batches(
            sentences, vocab, batch_size, shuffle=False):
        word_ids = word_ids.to(device); char_ids = char_ids.to(device)
        mask = mask.to(device)
        logits = model(word_ids, char_ids, mask)
        preds = logits.argmax(dim=-1)
        m = mask.cpu()
        p = preds.cpu(); t = tag_ids
        for bi in range(t.size(0)):
            for ti in range(t.size(1)):
                if m[bi, ti]:
                    y_true.append(int(t[bi, ti]))
                    y_pred.append(int(p[bi, ti]))
    return compute_metrics(y_true, y_pred, vocab.idx2tag), (y_true, y_pred)


def train_model(model, train_data: List[Sentence], dev_data: List[Sentence],
                vocab: Vocab, cfg: TrainConfig, device):
    set_seed(cfg.seed)
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                 weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2)

    best_macro = -1.0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "dev_acc": [], "dev_macro_f1": [],
               "dev_micro_f1": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for word_ids, char_ids, tag_ids, mask in make_batches(
                train_data, vocab, cfg.batch_size, shuffle=True,
                seed=cfg.seed + epoch):
            word_ids = word_ids.to(device); char_ids = char_ids.to(device)
            tag_ids = tag_ids.to(device); mask = mask.to(device)

            optimizer.zero_grad()
            logits = model(word_ids, char_ids, mask)
            loss = criterion(logits.view(-1, logits.size(-1)),
                             tag_ids.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        dev_metrics, _ = evaluate(model, dev_data, vocab, device)
        avg_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)
        history["dev_acc"].append(dev_metrics["accuracy"])
        history["dev_macro_f1"].append(dev_metrics["macro"]["f1"])
        history["dev_micro_f1"].append(dev_metrics["micro"]["f1"])
        scheduler.step(dev_metrics["macro"]["f1"])

        print(f"epoch {epoch:02d}  loss={avg_loss:.4f}  "
              f"dev {metrics_summary(dev_metrics)}")

        if dev_metrics["macro"]["f1"] > best_macro:
            best_macro = dev_metrics["macro"]["f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"early stopping at epoch {epoch} "
                      f"(best dev macro-F1={best_macro:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
