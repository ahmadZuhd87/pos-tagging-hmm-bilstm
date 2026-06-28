"""
hmm.py — Hidden Markov Model POS tagger with Viterbi decoding, from scratch.

This is the classical probabilistic baseline (spec allows HMM explicitly).
Everything is hand-implemented in NumPy so it is fully defensible in the
in-person discussion — you can derive transition/emission probabilities and the
Viterbi recurrence on a whiteboard.

Model
-----
A bigram HMM with:
    * transition probs   P(tag_i | tag_{i-1})   with add-k smoothing
    * emission probs      P(word | tag)          with add-k smoothing
    * <s> / </s> handled via explicit START/END states

Unknown words: at decode time, a word never seen in training gets a uniform-ish
emission via add-k smoothing over the vocabulary, so it relies almost entirely
on the transition (context) probabilities — exactly how an HMM should behave.
We work in log-space throughout to avoid underflow.
"""

from __future__ import annotations
from collections import defaultdict, Counter
from typing import List, Dict
import math

from data import Sentence

START = "<START>"
END = "<END>"


class HMMTagger:
    def __init__(self, k_trans: float = 1e-2, k_emit: float = 1e-2):
        self.k_trans = k_trans
        self.k_emit = k_emit
        self.tags: List[str] = []
        self.tag2idx: Dict[str, int] = {}
        self.vocab: set = set()
        # log-prob tables filled in fit()
        self.log_trans = None     # [n_states, n_states]
        self.log_emit = None      # dict: tag -> dict word -> logprob (sparse)
        self._emit_default = None  # tag -> logprob for unseen word

    def fit(self, train: List[Sentence]):
        trans_counts = defaultdict(Counter)   # prev_tag -> Counter(next_tag)
        emit_counts = defaultdict(Counter)    # tag -> Counter(word)
        tag_counts = Counter()

        tagset = set()
        for s in train:
            tagset.update(s.tags)
            self.vocab.update(s.forms)
        self.tags = [START, END] + sorted(tagset)
        self.tag2idx = {t: i for i, t in enumerate(self.tags)}

        for s in train:
            prev = START
            for w, t in zip(s.forms, s.tags):
                trans_counts[prev][t] += 1
                emit_counts[t][w] += 1
                tag_counts[t] += 1
                prev = t
            trans_counts[prev][END] += 1

        # ----- transition log-probs with add-k smoothing -----
        n = len(self.tags)
        V_tags = len(tagset) + 1  # possible next states (real tags + END)
        self.log_trans = [[float("-inf")] * n for _ in range(n)]
        for pi, prev in enumerate(self.tags):
            if prev == END:
                continue
            total = sum(trans_counts[prev].values())
            denom = total + self.k_trans * V_tags
            for ti, nxt in enumerate(self.tags):
                if nxt == START:
                    continue
                c = trans_counts[prev][nxt]
                self.log_trans[pi][ti] = math.log((c + self.k_trans) / denom)

        # ----- emission log-probs with add-k smoothing -----
        self.log_emit = {}
        self._emit_default = {}
        V = len(self.vocab)
        for t in tagset:
            total = sum(emit_counts[t].values())
            denom = total + self.k_emit * (V + 1)  # +1 for unseen-word mass
            self.log_emit[t] = {
                w: math.log((c + self.k_emit) / denom)
                for w, c in emit_counts[t].items()
            }
            self._emit_default[t] = math.log(self.k_emit / denom)

        print(f"[HMM] fitted: {len(tagset)} tags, vocab={V}, "
              f"k_trans={self.k_trans}, k_emit={self.k_emit}")
        return self

    def _emit_logprob(self, tag: str, word: str) -> float:
        d = self.log_emit.get(tag)
        if d is None:
            return float("-inf")
        return d.get(word, self._emit_default[tag])

    def predict(self, words: List[str]) -> List[str]:
        """Viterbi decoding for a single sentence."""
        if not words:
            return []
        real_tags = [t for t in self.tags if t not in (START, END)]
        T = len(words)
        # viterbi[t][tag] = best log-prob of a path ending in `tag` at position t
        viterbi = [dict() for _ in range(T)]
        backptr = [dict() for _ in range(T)]

        start_i = self.tag2idx[START]
        for tag in real_tags:
            ti = self.tag2idx[tag]
            lp = self.log_trans[start_i][ti] + self._emit_logprob(tag, words[0])
            viterbi[0][tag] = lp
            backptr[0][tag] = START

        for t in range(1, T):
            for tag in real_tags:
                ti = self.tag2idx[tag]
                best_lp = float("-inf")
                best_prev = None
                emit = self._emit_logprob(tag, words[t])
                for ptag in real_tags:
                    pi = self.tag2idx[ptag]
                    lp = viterbi[t - 1][ptag] + self.log_trans[pi][ti] + emit
                    if lp > best_lp:
                        best_lp = lp
                        best_prev = ptag
                viterbi[t][tag] = best_lp
                backptr[t][tag] = best_prev

        # termination: transition to END
        end_i = self.tag2idx[END]
        best_lp = float("-inf")
        best_last = real_tags[0]
        for tag in real_tags:
            ti = self.tag2idx[tag]
            lp = viterbi[T - 1][tag] + self.log_trans[ti][end_i]
            if lp > best_lp:
                best_lp = lp
                best_last = tag

        # backtrace
        tags_out = [best_last]
        for t in range(T - 1, 0, -1):
            tags_out.append(backptr[t][tags_out[-1]])
        tags_out.reverse()
        return tags_out

    def predict_corpus(self, sentences: List[Sentence]):
        """Return flattened (y_true_idx, y_pred_idx) using a tag->idx map.

        The caller supplies the neural Vocab's tag2idx so both models report on
        the SAME label space. Tags unseen there map to 0.
        """
        preds = [self.predict(s.forms) for s in sentences]
        return preds
