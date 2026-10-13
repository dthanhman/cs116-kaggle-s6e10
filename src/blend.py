"""Blend / stack nhiều model trên OOF.

Chạy: python src/blend.py lgb mlp                     -> AUC từng model, tương quan, rank-blend đều
      python src/blend.py lgb mlp --weights 2 1       -> rank-blend có trọng số
      python src/blend.py --method lr --C 0.003       -> stack LR trên mọi oof_*.npy trong OOF_DIR
      python src/blend.py --method lr --meta-seed 7 101  -> đối chiếu stack trên meta fold seed khác
      thêm --save <tên> để lưu oof_<tên>.npy, test_<tên>.npy

Không ghi tên model thì lấy mọi oof_*.npy trong OOF_DIR, trừ file bắt đầu bằng blend_ / stack_
(để không đưa bản blend cũ vào lại). Nên đặt --save là blend_<...> hoặc stack_<...>.

Stack LR dùng logit của rank (mỗi member đổi về rank trước), nên member lưu dạng xác suất,
logit hay rank đều dùng được như nhau. Mặc định meta fold = folds.csv; --meta-seed chỉ để
kiểm tra độ ổn định, chọn cấu hình theo kết quả ổn định qua nhiều seed.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import DATA_DIR, ID, N_FOLDS, OOF_DIR, TARGET
from utils import load_preds, save_preds

EXCLUDE_PREFIXES = ("blend_", "stack_")


def to_rank(p):
    return rankdata(p) / len(p)


def to_logit_rank(p):
    r = rankdata(p) / (len(p) + 1)
    return np.log(r / (1 - r)).astype(np.float32)


def rank_blend(preds, weights=None):
    weights = np.ones(len(preds)) if weights is None else np.asarray(weights, dtype=float)
    return sum(w * to_rank(p) for w, p in zip(weights, preds)) / weights.sum()


def lr_stack(X, y, X_test, folds, C=0.003):
    """Stack LR: mỗi fold k train trên các fold còn lại, dự đoán fold k; test = trung bình các fold."""
    oof, test_pred, coefs = np.zeros(len(y)), np.zeros(len(X_test)), []
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        m = LogisticRegression(C=C, max_iter=2000).fit(X[tr], y[tr])
        oof[va] = m.decision_function(X[va])
        test_pred += m.decision_function(X_test) / len(np.unique(folds))
        coefs.append(m.coef_[0])
    return oof, test_pred, np.mean(coefs, axis=0)


def load_y_folds():
    train = pd.read_csv(DATA_DIR / "train.csv", usecols=[ID, TARGET])
    folds = pd.read_csv(DATA_DIR / "folds.csv")
    assert folds[ID].equals(train[ID]), "folds.csv không khớp thứ tự id của train.csv"
    return train[TARGET].astype(int).values, folds["fold"].values


def meta_folds(y, seed):
    folds = np.zeros(len(y), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(y, y)):
        folds[va] = k
    return folds


def all_names():
    names = sorted(p.stem[len("oof_"):] for p in OOF_DIR.glob("oof_*.npy"))
    return [n for n in names if not n.startswith(EXCLUDE_PREFIXES)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="bỏ trống = mọi oof_*.npy trong OOF_DIR")
    ap.add_argument("--method", choices=["rank", "lr"], default="rank")
    ap.add_argument("--weights", nargs="+", type=float, help="chỉ dùng với --method rank")
    ap.add_argument("--C", type=float, default=0.003, help="regularization của stack LR")
    ap.add_argument("--meta-seed", nargs="+", type=int, help="đối chiếu stack LR trên meta fold seed khác")
    ap.add_argument("--save")
    args = ap.parse_args()

    names = args.names or all_names()
    assert names, f"không tìm thấy oof_*.npy nào trong {OOF_DIR}"
    y, folds = load_y_folds()
    oofs, tests = zip(*(load_preds(n) for n in names))

    print(f"{len(names)} member")
    for n, o in sorted(zip(names, oofs), key=lambda t: -roc_auc_score(y, t[1])):
        print(f"  {n:<30} CV AUC = {roc_auc_score(y, o):.6f}")
    if 1 < len(names) <= 15:
        ranks = np.column_stack([to_rank(o) for o in oofs])
        print("\nTương quan rank giữa các OOF:")
        print(pd.DataFrame(np.corrcoef(ranks, rowvar=False), index=names, columns=names).round(4))

    if args.method == "rank":
        oof_b, test_b = rank_blend(oofs, args.weights), rank_blend(tests, args.weights)
        print(f"\n{'RANK BLEND':<32} CV AUC = {roc_auc_score(y, oof_b):.6f}")
    else:
        X = np.column_stack([to_logit_rank(o) for o in oofs])
        X_test = np.column_stack([to_logit_rank(t) for t in tests])
        oof_b, test_b, coef = lr_stack(X, y, X_test, folds, args.C)
        print(f"\n{f'LR STACK C={args.C}':<32} CV AUC = {roc_auc_score(y, oof_b):.6f} (meta fold = folds.csv)")
        for s in args.meta_seed or []:
            o, _, _ = lr_stack(X, y, X_test, meta_folds(y, s), args.C)
            print(f"{'':<32} CV AUC = {roc_auc_score(y, o):.6f} (meta fold seed {s})")
        top = np.argsort(-np.abs(coef))[:15]
        print("\nHệ số lớn nhất:")
        for i in top:
            print(f"  {names[i]:<30} {coef[i]:+.4f}")

    if args.save:
        save_preds(args.save, oof_b, test_b, y)
