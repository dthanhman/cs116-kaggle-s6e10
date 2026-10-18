"""
CS116 LLR → OOF Pipeline (chạy 1 lệnh là xong)
=================================================
Chạy:   python run_llr_pipeline.py
Smoke:  SMOKE=1 python run_llr_pipeline.py   (chạy nhanh để test, 60 cây thay 6000)

Script này thực hiện toàn bộ pipeline của notebook cs116_llr_oof_local.ipynb:
  1. Kiểm tra files cần thiết
  2. Load data + LLR features
  3. LogReg baseline trên LLR (đánh giá tín hiệu thuần của LLR)
  4. XGB 5-fold CV trên features thô + LLR
  5. Lưu OOF sạch vào oof/
  6. Verify bằng blend.py
  7. Tạo file CSV nộp Kaggle vào submissions/

Đầu ra:
  oof/oof_cs116_llr_xgb_v1.npy     (OOF predictions)
  oof/test_cs116_llr_xgb_v1.npy    (test predictions)
  submissions/sub_cs116_llr_xgb_v1.csv   (file nộp Kaggle)
"""

import sys
import os
import time
import subprocess
import warnings

# ── Thêm src/ vào path ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")



import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, OOF_DIR, SUB_DIR, N_FOLDS, SEED, TARGET, ID, CAT_COLS, FEATURES
from data import load_data
from utils import save_preds, N_TRAIN, N_TEST

warnings.filterwarnings("ignore")

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


# ╔══════════════════════════════════════════════════════════╗
# ║                    CẤU HÌNH                              ║
# ╚══════════════════════════════════════════════════════════╝
LLR_DIR = os.environ.get("LLR_DIR", str(DATA_DIR / "llr"))
SMOKE = os.environ.get("SMOKE") == "1"

NAME = "cs116_llr_xgb_v1"
PERSON = "?"
REFIT = False
RUN_ABLATION = False
MAX_TREES, ES_ROUNDS = (60, 10) if SMOKE else (6000, 200)
MIN_AUC, MAX_AUC = 0.90, 0.962

assert not NAME.startswith(("blend_", "stack_", "team_"))

XGB_PARAMS = dict(
    learning_rate=0.05, max_depth=6, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0,
    tree_method="hist", enable_categorical=True, eval_metric="auc",
    random_state=SEED, n_jobs=os.cpu_count() or 4,
)

try:
    subprocess.check_output("nvidia-smi", stderr=subprocess.DEVNULL)
    DEV = "cuda"
except Exception:
    DEV = "cpu"

log("xgboost", xgb.__version__, "| device", DEV, "| SMOKE", SMOKE)

# ╔══════════════════════════════════════════════════════════╗
# ║  1. KIỂM TRA FILES                                      ║
# ╚══════════════════════════════════════════════════════════╝
log("=== Bước 1: Kiểm tra files ===")

required_data = [DATA_DIR / "train.csv", DATA_DIR / "test.csv", DATA_DIR / "folds.csv"]
required_llr = [
    "llr_train.npy", "llr_test.npy",
    "lp_true_train.npy", "lp_true_test.npy",
    "lp_false_train.npy", "lp_false_test.npy",
    "llr_orders_train.npy", "llr_orders_test.npy",
]

missing = []
for p in required_data:
    if not p.exists():
        missing.append(str(p))
for f in required_llr:
    if not os.path.exists(os.path.join(LLR_DIR, f)):
        missing.append(os.path.join(LLR_DIR, f))

if missing:
    print("❌ THIẾU FILE:")
    for m in missing:
        print(f"   {m}")
    print("\nXem README.md mục 1 để tải data, và đảm bảo data/llr/ có output của cs116_llr_score.py")
    sys.exit(1)

log("Tất cả files OK")

# ╔══════════════════════════════════════════════════════════╗
# ║  2. LOAD DATA + LLR FEATURES                            ║
# ╚══════════════════════════════════════════════════════════╝
log("=== Bước 2: Load data ===")

train, test = load_data()
y = train[TARGET].to_numpy()
fold = train["fold"].to_numpy()
N, NT = len(train), len(test)
log(f"train {N:,} dòng | test {NT:,} dòng")
log(f"số dòng mỗi fold: {np.bincount(fold).tolist()}")


def find_npy(fname):
    p = os.path.join(LLR_DIR, fname)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}: cần output của cs116_llr_score.py")
    return p


def llr_frame(split):
    g = lambda n: np.load(find_npy(f"{n}_{split}.npy"))
    orders = g("llr_orders")
    df = pd.DataFrame({"llr": g("llr"), "lp_true": g("lp_true"), "lp_false": g("lp_false")})
    for k in range(orders.shape[1]):
        df[f"llr_o{k}"] = orders[:, k]
    if orders.shape[1] >= 2:
        df["llr_gap"] = orders[:, 0] - orders[:, 1]
    return df


LLR_TR, LLR_TE = llr_frame("train"), llr_frame("test")
assert len(LLR_TR) == N and len(LLR_TE) == NT
assert np.isfinite(LLR_TR.to_numpy()).all() and np.isfinite(LLR_TE.to_numpy()).all(), "LLR có NaN/inf"
LLR_COLS = list(LLR_TR.columns)

auc_llr_raw = roc_auc_score(y, LLR_TR["llr"])
fold_auc = [roc_auc_score(y[fold == k], LLR_TR["llr"][fold == k]) for k in range(N_FOLDS)]
log(f"AUC LLR thô: {auc_llr_raw:.6f} | fold: {np.round(fold_auc, 5).tolist()}")
assert auc_llr_raw > 0.6 or SMOKE, "LLR không có tín hiệu — có thể lệch thứ tự dòng"

for c in ("llr", "llr_gap"):
    if c in LLR_TR:
        d = abs(LLR_TE[c].mean() - LLR_TR[c].mean()) / LLR_TR[c].std()
        log(f"drift {c}: {d:.4f}" + ("   <-- đáng ngờ" if d > 0.1 else ""))

# ╔══════════════════════════════════════════════════════════╗
# ║  3. LOGREG BASELINE (chỉ LLR features)                  ║
# ╚══════════════════════════════════════════════════════════╝
log("=== Bước 3: LogReg baseline ===")


def cv_logreg(cols):
    oof, tst = np.zeros(N), np.zeros(NT)
    for k in range(N_FOLDS):
        tr, va = fold != k, fold == k
        sc = StandardScaler().fit(LLR_TR.loc[tr, cols])
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(LLR_TR.loc[tr, cols]), y[tr])
        oof[va] = lr.predict_proba(sc.transform(LLR_TR.loc[va, cols]))[:, 1]
        tst += lr.predict_proba(sc.transform(LLR_TE[cols]))[:, 1] / N_FOLDS
    return oof, tst


oof_lr, tst_lr = cv_logreg(LLR_COLS)
cv_lr = roc_auc_score(y, oof_lr)
log(f"LogReg chỉ LLR features: CV AUC {cv_lr:.6f}")

# ╔══════════════════════════════════════════════════════════╗
# ║  4. XGB + LLR (5-fold CV)                               ║
# ╚══════════════════════════════════════════════════════════╝
log("=== Bước 4: XGB raw + LLR ===")

# Kiểm tra FEATURES khớp cột thực tế
expected = set(train.columns) - {ID, TARGET, "fold"}
assert set(FEATURES) == expected, f"FEATURES thiếu/thừa: {set(FEATURES) ^ expected}"

X_base, XT_base = train[FEATURES].copy(), test[FEATURES].copy()
for c in CAT_COLS:
    cats = sorted(set(X_base[c].dropna().astype(str)) | set(XT_base[c].dropna().astype(str)))
    for d in (X_base, XT_base):
        d[c] = pd.Categorical(d[c].astype("string"), categories=cats)

X_llr = pd.concat([X_base, LLR_TR], axis=1)
XT_llr = pd.concat([XT_base, LLR_TE], axis=1)
log(f"{len(FEATURES)} cột thô ({len(CAT_COLS)} category) + {len(LLR_COLS)} cột LLR: {LLR_COLS}")


def make_clf(n_estimators, early=False):
    kw = dict(XGB_PARAMS, device=DEV, n_estimators=n_estimators)
    if early:
        kw["early_stopping_rounds"] = ES_ROUNDS
    return xgb.XGBClassifier(**kw)


def run_cv(X, XT, want_test, tag):
    oof = np.zeros(N)
    tst = np.zeros(len(XT)) if want_test else None
    aucs, iters = [], []
    for k in range(N_FOLDS):
        t0 = time.time()
        tr_idx, va_idx = np.where(fold != k)[0], np.where(fold == k)[0]
        rng = np.random.default_rng(42)
        es_idx = rng.choice(tr_idx, size=len(tr_idx) // 10, replace=False)
        fit_idx = np.setdiff1d(tr_idx, es_idx)
        m = make_clf(MAX_TREES, early=True).fit(
            X.iloc[fit_idx], y[fit_idx],
            eval_set=[(X.iloc[es_idx], y[es_idx])], verbose=False,
        )
        best = int(m.best_iteration) + 1
        n_use = best
        if REFIT:
            n_use = max(10, int(best * 1.1))
            m = make_clf(n_use).fit(X.iloc[tr_idx], y[tr_idx], verbose=False)
        oof[va_idx] = m.predict_proba(X.iloc[va_idx], iteration_range=(0, n_use))[:, 1]
        if want_test:
            tst += m.predict_proba(XT, iteration_range=(0, n_use))[:, 1] / N_FOLDS
        aucs.append(roc_auc_score(y[va_idx], oof[va_idx]))
        iters.append(best)
        log(f"[{tag}] fold {k}: AUC {aucs[-1]:.6f} | {best} cây | {time.time() - t0:.0f}s")
    return oof, tst, np.array(aucs), iters


oof_x, tst_x, auc_x, it_x = run_cv(X_llr, XT_llr, True, "raw+LLR")
cv_x = roc_auc_score(y, oof_x)
log(f"XGB raw+LLR: CV AUC {cv_x:.6f} | fold-mean {auc_x.mean():.6f} ± {auc_x.std():.6f} | cây {it_x}")

if RUN_ABLATION:
    _, _, auc_b, it_b = run_cv(X_base, XT_base, False, "raw (không LLR)")
    d = auc_x - auc_b
    log(f"LLR giúp {int((d > 0).sum())}/{N_FOLDS} fold | trung bình {d.mean() * 1e6:+.1f}e-6")

# ╔══════════════════════════════════════════════════════════╗
# ║  5. LƯU OOF + KIỂM TRA                                 ║
# ╚══════════════════════════════════════════════════════════╝
log("=== Bước 5: Lưu OOF ===")


def check_and_save(name, oof, tst, y_true):
    assert not name.startswith(("blend_", "stack_", "team_"))
    assert np.isfinite(oof).all() and np.isfinite(tst).all(), "NaN/inf"
    auc = roc_auc_score(y_true, oof)
    if auc < MIN_AUC:
        status = f"QUÁ THẤP (<{MIN_AUC})"
    elif auc > MAX_AUC:
        status = f"QUÁ CAO: nghi leak (>{MAX_AUC})"
    else:
        status = "OK"
    log(f"{name}: CV AUC {auc:.6f} [{status}] | OOF mean {oof.mean():.4f} test mean {tst.mean():.4f}")
    save_preds(name, oof, tst)
    return auc, status


auc_main, st_main = check_and_save(NAME, oof_x, tst_x, y)

if MIN_AUC <= cv_lr <= MAX_AUC:
    lr_name = NAME.replace("xgb", "lr")
    check_and_save(lr_name, oof_lr, tst_lr, y)

# ╔══════════════════════════════════════════════════════════╗
# ║  6. VERIFY BẰNG BLEND.PY                                ║
# ╚══════════════════════════════════════════════════════════╝
log("=== Bước 6: Verify ===")

try:
    from utils import load_preds
    o, t = load_preds(NAME)
    verify_auc = roc_auc_score(y, o)
    log(f"Verify load_preds('{NAME}'): AUC {verify_auc:.6f} — {'OK' if abs(verify_auc - auc_main) < 1e-9 else 'SAI!'}")
except Exception as e:
    log(f"Verify failed: {e}")

# ╔══════════════════════════════════════════════════════════╗
# ║  7. TẠO FILE CSV NỘP KAGGLE                             ║
# ╚══════════════════════════════════════════════════════════╝
log("=== Bước 7: Tạo file nộp Kaggle ===")

pred = np.load(OOF_DIR / f"test_{NAME}.npy")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
assert len(pred) == len(sample), f"số dòng {len(pred)} khác sample_submission {len(sample)}"
assert not np.isnan(pred).any(), "có NaN"
assert np.unique(pred).size > 2, "có vẻ là nhãn 0/1, phải nộp xác suất"

sub = pd.DataFrame({ID: sample[ID], TARGET: pred})
SUB_DIR.mkdir(exist_ok=True)
out_csv = SUB_DIR / f"sub_{NAME}.csv"
sub.to_csv(out_csv, index=False)
log(f"Đã lưu {out_csv}")
print(sub.head())

# ╔══════════════════════════════════════════════════════════╗
# ║  8. TỔNG KẾT                                            ║
# ╚══════════════════════════════════════════════════════════╝
import datetime

print("\n" + "=" * 60)
print("TỔNG KẾT")
print("=" * 60)
print(f"  Model:          {NAME}")
print(f"  CV AUC:         {auc_main:.6f} [{st_main}]")
print(f"  Fold AUCs:      {np.round(auc_x, 6).tolist()}")
print(f"  Số cây/fold:    {it_x}")
print(f"  Device:         {DEV}")
print(f"  Thời gian:      {time.time() - T0:.0f}s")
print()
print("Đầu ra:")
print(f"  OOF:        {OOF_DIR / f'oof_{NAME}.npy'}")
print(f"  Test:       {OOF_DIR / f'test_{NAME}.npy'}")
print(f"  Submission: {out_csv}")
print()
print("Dòng cho experiments.csv:")
print(f"  {datetime.date.today()},{PERSON},{NAME},{auc_main:.6f},,XGB cot tho + LLR distilgpt2; "
      f"ES 10% trong fold, REFIT={REFIT}; fold-mean {auc_x.mean():.6f}")
print()
print(f'Nộp Kaggle:\n  kaggle competitions submit -c playground-series-s6e10 -f "{out_csv}" -m "{NAME}"')
print("=" * 60)
