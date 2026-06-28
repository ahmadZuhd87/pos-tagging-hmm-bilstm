# ENCS5342 — POS Tagging (Track 1)

Part-of-Speech tagging on the English Penn Treebank, predicting the **joint
coarse–fine tag** (e.g. `NOUN-NNP`). Two from-scratch models are built and
compared: an **HMM + Viterbi** baseline and a **BiLSTM + character-CNN** neural
tagger.

- **Course:** ENCS5342 — Information Retrieval with Applications of NLP (Term 1252)
- **Track:** 1 — Default Project (POS Tagging)
- **Team:** Ahmad Zuhd - 1222332, Bara Mohsen - 1220829

## Repository structure
```
.
├── notebooks/
│   └── ENCS5342_POS_Tagging.ipynb   # main deliverable (all 10 required sections)
├── src/
│   ├── data.py        # CoNLL parser, Vocab, batching  (from scratch)
│   ├── hmm.py         # HMM + Viterbi tagger           (from scratch)
│   ├── bilstm.py      # BiLSTM + char-CNN model         (from scratch)
│   ├── train.py       # training loop, early stopping, LR sched, seeding
│   ├── metrics.py     # per-class / micro / macro P-R-F1, confusion  (from scratch)
│   └── ablations.py   # ablation experiment harness
├── requirements.txt
└── README.md
```

## Getting the data
The dataset is **not** committed (copyrighted; do not redistribute). Download
`en-universal-train.conll` and `en-universal-dev.conll` from the course Google
Drive folder and place them in the project root.

## Running
```bash
pip install -r requirements.txt
# open the notebook and run top-to-bottom:
jupyter notebook notebooks/ENCS5342_POS_Tagging.ipynb
# or run the ablation study from the CLI:
python src/ablations.py --train en-universal-train.conll --dev en-universal-dev.conll
```

All randomness is seeded (`seed=42`) for reproducibility.

## Notes
- Models are implemented **from scratch** (no high-level token-classification
  pipeline) — this is a graded requirement of Track 1.
- The held-out test set is evaluated by the instructor after submission.
