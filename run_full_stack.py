"""
CS116 Kaggle S6E10 - Full Team Stacking Pipeline
================================================
Kết hợp cả 3 họ mô hình mạnh nhất của nhóm:
  1. GBDT Họ Cây Nâng Cao:
     - quan_cat_dual_teacher_v4 (CatBoost GPU Dual Teacher - CV 0.96094)
     - quan_xgb_dual_teacher_v4 (XGBoost GPU Dual Teacher - CV 0.96092)
     - quan_xgb_safe_teacher_v4 (XGBoost GPU Safe Teacher - CV 0.96086)
     - quan_xgb_original_aug_v4 (XGBoost Original Augmentation - CV 0.95853)
  2. Mạng Nơ-ron Deep Learning:
     - A_tabm_v1 (TabM Batch Ensemble ICLR 2025 - CV 0.95954)
  3. Mô Hình Ngôn Ngữ Sinh Dữ Liệu:
     - cs116_llr_xgb_v1 (distilgpt2 Log-Likelihood Ratio Surrogate - CV 0.95831)

Huấn luyện Meta-Model (Logistic Regression trên thang đo Logit) để tìm trọng số tối ưu.
Sau đó kết hợp với mốc tham chiếu P247 (0.96183 LB) để tạo ra file nộp bài tối thượng.

Chạy: python run_full_stack.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ── Đảm bảo hiển thị tiếng Việt mượt mà trên console Windows ──
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── Thêm src/ vào sys.path ──
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import DATA_DIR, OOF_DIR, SUB_DIR, TARGET, ID
from utils import load_preds, save_preds
from blend import to_logit, load_y_folds

warnings.filterwarnings("ignore")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


print("=" * 65)
print("     CS116 KAGGLE S6E10 - FULL TEAM STACKING PIPELINE")
print("=" * 65)

# 1. Nạp nhãn và fold chuẩn
log("1. Nạp nhãn và fold chung (folds.csv)...")
y_true, folds = load_y_folds()
N_TRAIN = len(y_true)

# 2. Danh sách các model chủ lực của cả 3 họ
TEAM_MEMBERS = [
    ("quan_cat_dual_teacher_v4", "GBDT (CatBoost Dual Teacher)"),
    ("quan_xgb_dual_teacher_v4", "GBDT (XGBoost Dual Teacher)"),
    ("quan_xgb_safe_teacher_v4", "GBDT (XGBoost Safe Teacher)"),
    ("quan_xgb_original_aug_v4", "GBDT (XGBoost Original Aug)"),
    ("A_tabm_v1",                 "Deep Learning (TabM Batch Ensemble)"),
    ("cs116_llr_xgb_v1",         "GenAI (distilgpt2 LLR Surrogate)"),
]

log(f"2. Nạp OOF và Test predictions của {len(TEAM_MEMBERS)} mô hình chủ lực...")
oofs, tests, names = [], [], []

for name, family in TEAM_MEMBERS:
    try:
        oof_p, test_p = load_preds(name)
        auc = roc_auc_score(y_true, oof_p)
        oofs.append(oof_p)
        tests.append(test_p)
        names.append(name)
        log(f"   ✓ {name:<26} [{family}]: CV AUC = {auc:.6f}")
    except FileNotFoundError:
        log(f"   ✗ Không tìm thấy file {name} trong oof/ (bỏ qua)")

assert len(names) >= 2, "Cần ít nhất 2 model để thực hiện stacking!"

# 3. Tính ma trận tương quan rank giữa các họ
log("\n3. Ma trận tương quan rank giữa các mô hình (Diversity Matrix):")
R = np.column_stack([rankdata(o) / len(o) for o in oofs])
corr_df = pd.DataFrame(np.corrcoef(R, rowvar=False), index=names, columns=names)
print(corr_df.round(4).to_string())

# 4. Huấn luyện Meta Stacker (Logistic Regression trên thang đo logit)
log("\n4. Huấn luyện Meta-Model (Logistic Regression Stacker)...")
Z = [to_logit(o, t) for o, t in zip(oofs, tests)]
X = np.column_stack([z[0] for z in Z])
X_test = np.column_stack([z[1] for z in Z])

# Tìm regularization tối ưu nhẹ
oof_stack = np.zeros(N_TRAIN)
test_stack = np.zeros(len(tests[0]))
coefs = []

C_VALUE = 0.003
for k in np.unique(folds):
    tr, va = folds != k, folds == k
    m = LogisticRegression(C=C_VALUE, tol=1e-5, max_iter=2000).fit(X[tr], y_true[tr])
    oof_stack[va] = m.decision_function(X[va])
    test_stack += m.decision_function(X_test) / len(np.unique(folds))
    coefs.append(m.coef_[0])

# Đổi margin về xác suất (sigmoid)
oof_stack_prob = 1 / (1 + np.exp(-oof_stack))
test_stack_prob = 1 / (1 + np.exp(-test_stack))
cv_stack = roc_auc_score(y_true, oof_stack_prob)

log(f"\n   ★ CV AUC của Team Stacker = {cv_stack:.6f} ★")

# Trọng số của từng họ mô hình
avg_coef = np.mean(coefs, axis=0)
order = np.argsort(-np.abs(avg_coef))
log("\n   Trọng số đóng góp của từng mô hình (Weights):")
for idx in order:
    print(f"      {names[idx]:<28} : {avg_coef[idx]:+.4f}")

# Lưu OOF & Test của Team Stack
STACK_NAME = "team_stack_v2"
save_preds(STACK_NAME, oof_stack_prob, test_stack_prob, y_true)

# 5. Xuất file nộp bài của riêng Team Stack
SUB_DIR.mkdir(exist_ok=True)
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

sub_team = pd.DataFrame({ID: sample[ID], TARGET: test_stack_prob})
out_team_csv = SUB_DIR / f"sub_{STACK_NAME}.csv"
sub_team.to_csv(out_team_csv, index=False)
log(f"\n5. Đã lưu bài nộp thuần của nhóm: {out_team_csv}")

# 6. Kết hợp với mốc tham chiếu P247 (0.96183 LB)
REF_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_score", "sub_track3_fw_p247_x_96183_w03.csv")
if os.path.exists(REF_CSV):
    log("\n6. Tìm thấy file tham chiếu P247 (LB 0.96183) trong ref_score/!")
    ref_df = pd.read_csv(REF_CSV)
    
    # Rank blend chuẩn hóa [0, 1]
    r_ref = rankdata(ref_df["satisfaction"]) / (len(ref_df) + 1)
    r_team = rankdata(test_stack_prob) / (len(test_stack_prob) + 1)
    
    # Tương quan giữa Team Stack và P247
    corr_with_ref = np.corrcoef(r_ref, r_team)[0, 1]
    log(f"   Tương quan rank giữa Team Stack và P247 Benchmark: {corr_with_ref:.4f}")
    
    # Tạo 2 phương án nộp an toàn:
    # Blend 1: Bảo toàn mốc 0.96183 + 5% tín hiệu độc quyền của nhóm
    blend_w05 = 0.95 * r_ref + 0.05 * r_team
    sub_blend_w05 = pd.DataFrame({ID: sample[ID], TARGET: rankdata(blend_w05) / (len(blend_w05) + 1)})
    out_blend_w05 = SUB_DIR / "sub_final_p247_team_blend_w05.csv"
    sub_blend_w05.to_csv(out_blend_w05, index=False)
    log(f"   ✓ Đã xuất bài nộp an toàn (95% P247 + 5% Team): {out_blend_w05}")

    # Blend 2: Cân bằng (85% P247 + 15% Team)
    blend_w15 = 0.85 * r_ref + 0.15 * r_team
    sub_blend_w15 = pd.DataFrame({ID: sample[ID], TARGET: rankdata(blend_w15) / (len(blend_w15) + 1)})
    out_blend_w15 = SUB_DIR / "sub_final_p247_team_blend_w15.csv"
    sub_blend_w15.to_csv(out_blend_w15, index=False)
    log(f"   ✓ Đã xuất bài nộp đột phá (85% P247 + 15% Team): {out_blend_w15}")

print("\n" + "=" * 65)
print("                    TỔNG KẾT BÀI NỘP")
print("=" * 65)
print(f"1. Bài nộp thuần nhóm:      {out_team_csv}")
if os.path.exists(REF_CSV):
    print(f"2. Bài nộp tối thượng (5%):  {out_blend_w05} (KHUYẾN NGHỊ NỘP NGAY)")
    print(f"3. Bài nộp đột phá (15%):    {out_blend_w15}")
print("=" * 65)
