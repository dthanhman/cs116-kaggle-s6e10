"""Rank-blend nhiều model trên OOF (bản tối thiểu, C mở rộng thêm tối ưu trọng số / stacking).

Chạy: python src/blend.py lgb mlp           -> in AUC từng model và AUC blend
      python src/blend.py lgb mlp --save v1 -> lưu thêm oof_v1.npy, test_v1.npy
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

from config import DATA_DIR, TARGET
from utils import load_preds, save_preds


def to_rank(p):
    return rankdata(p) / len(p)


def rank_blend(preds, weights=None):
    weights = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=float)
    return sum(w * to_rank(p) for w, p in zip(weights, preds)) / weights.sum()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--weights", nargs="+", type=float)
    ap.add_argument("--save")
    args = ap.parse_args()

    y = pd.read_csv(DATA_DIR / "train.csv", usecols=[TARGET])[TARGET].astype(int).values
    oofs, tests = zip(*(load_preds(n) for n in args.names))
    for n, o in zip(args.names, oofs):
        print(f"{n:<20} CV AUC = {roc_auc_score(y, o):.6f}")

    oof_b, test_b = rank_blend(oofs, args.weights), rank_blend(tests, args.weights)
    print(f"{'BLEND':<20} CV AUC = {roc_auc_score(y, oof_b):.6f}")
    if args.save:
        save_preds(args.save, oof_b, test_b, y)
