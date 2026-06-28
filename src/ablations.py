"""
ablations.py — Ablation experiments for the BiLSTM tagger.

Runs the model with single techniques removed/added so you can REPORT the effect
of each (the rubric and Tip #6 explicitly reward ablation experiments). Each row
in the printed table is a defensible experimental result.

Configurations compared (all else equal):
    full        : char-CNN + dropout + GloVe + early stop + LR sched
    no_char     : remove character encoder  (isolates UNK-word handling)
    no_dropout  : dropout = 0               (isolates regularisation)
    no_pretrain : random word embeddings    (isolates GloVe contribution)

Run after the data files are in place:
    python ablations.py --train ../en-universal-train.conll --dev ../en-universal-dev.conll
"""

from __future__ import annotations
import argparse
import torch

from data import parse_conll, Vocab
from bilstm import BiLSTMTagger, ModelConfig
from train import train_model, TrainConfig, evaluate
from metrics import metrics_summary


def run(train, dev, vocab, device, name, **overrides):
    cfg = ModelConfig(n_words=vocab.n_words, n_chars=vocab.n_chars,
                      n_tags=vocab.n_tags, **overrides)
    model = BiLSTMTagger(cfg)
    if cfg.use_pretrained and overrides.get("glove_path"):
        model.load_glove(overrides["glove_path"], vocab.idx2word)
    model, hist = train_model(model, train, dev, vocab,
                              TrainConfig(epochs=30, patience=5), device)
    m, _ = evaluate(model, dev, vocab, device)
    print(f"\n=== {name}: {metrics_summary(m)} ===\n")
    return name, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--glove", default=None,
                    help="optional path to glove.6B.100d.txt")
    ap.add_argument("--min_freq", type=int, default=2)
    args = ap.parse_args()

    train = parse_conll(args.train)
    dev = parse_conll(args.dev)
    vocab = Vocab.build(train, min_freq=args.min_freq)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    results = []
    results.append(run(train, dev, vocab, device, "full",
                       use_char=True, dropout=0.5,
                       use_pretrained=bool(args.glove),
                       glove_path=args.glove if args.glove else None))
    results.append(run(train, dev, vocab, device, "no_char", use_char=False))
    results.append(run(train, dev, vocab, device, "no_dropout",
                       use_char=True, dropout=0.0, emb_dropout=0.0))
    if args.glove:
        results.append(run(train, dev, vocab, device, "no_pretrain",
                           use_char=True, use_pretrained=False))

    print("\n\n================ ABLATION SUMMARY ================")
    print(f"{'config':<14}{'acc':>8}{'micro-F1':>10}{'macro-F1':>10}")
    for name, m in results:
        print(f"{name:<14}{m['accuracy']:>8.4f}"
              f"{m['micro']['f1']:>10.4f}{m['macro']['f1']:>10.4f}")


if __name__ == "__main__":
    main()
