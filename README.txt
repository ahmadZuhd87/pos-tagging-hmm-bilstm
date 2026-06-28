ENCS5342 Final Project — Track 1: POS Tagging
=============================================

HOW TO RUN
----------
1. Install dependencies:
       pip install -r requirements.txt
2. Download the data files from the course Google Drive folder and place them
   in the project root:
       en-universal-train.conll
       en-universal-dev.conll
   (The dataset is copyrighted and is NOT included in this package.)
3. Run the notebook top to bottom:
       jupyter notebook notebooks/ENCS5342_POS_Tagging.ipynb
   It produces all reported results (HMM + BiLSTM, train + dev metrics,
   per-class/micro/macro P-R-F1, learning curves, heatmaps, ablations).
4. (Optional) Run the ablation study from the command line:
       python src/ablations.py --train en-universal-train.conll \
                               --dev   en-universal-dev.conll

REPRODUCIBILITY
---------------
All random seeds are fixed to 42 (Python, NumPy, PyTorch). Running the notebook
from a clean kernel reproduces every reported number.

DEPENDENCIES (versions)
-----------------------
torch>=2.0, numpy>=1.24, pandas>=2.0, scikit-learn>=1.3,
matplotlib>=3.7, seaborn>=0.12, nbformat>=5.9, jupyter>=1.0

CONTENTS
--------
notebooks/ENCS5342_POS_Tagging.ipynb  - main notebook (10 required sections)
src/                                  - from-scratch model & utility modules
README.md                             - GitHub repo readme
requirements.txt                      - dependency list

OPTIONAL: pre-trained embeddings
--------------------------------
To enable GloVe initialisation, download glove.6B.100d.txt and call
model.load_glove("glove.6B.100d.txt", vocab.idx2word) in the BiLSTM cell.
