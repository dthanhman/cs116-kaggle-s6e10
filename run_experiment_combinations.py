"""
CS116 Kaggle S6E10 - Script Đánh Giá & Thử Nghiệm 3 Tổ Hợp Mô Hình
==================================================================
Thực hiện thử nghiệm khoa học theo yêu cầu:
  - Trường hợp 1 (Case 1): Baseline Team + Model 1 (B_personal_spec_v1 - Dual Specialist GBDT)
  - Trường hợp 2 (Case 2): Baseline Team + Model 2 (B_realmlp_v1 - RealMLP Neural Network)
  - Trường hợp 3 (Case 3): Baseline Team + CẢ HAI (B_personal_spec_v1 + B_realmlp_v1)

Quy trình mỗi trường hợp:
  1. Kiểm tra tính đa dạng (Diversity Rank Correlation).
  2. Huấn luyện Meta-Stacker (Logistic Regression trên Logit) qua 5 Fold chung (folds.csv).
  3. Đo lường chính xác CV AUC tổng thể, mức tăng delta-CV so với baseline 0.961175.
  4. Trích xuất trọng số (weights) của từng model trong tổ hợp.
  5. Tạo các file nộp bài kết hợp với mốc tham chiếu P247 (LB 0.96183) với 3 mức tỷ trọng:
     - w05: 5% Team + 95% P247 (An toàn)
     - w10: 10% Team + 90% P247 (Khuyến nghị)
     - w15: 15% Team + 85% P247 (Đột phá)

Chạy: python run_experiment_combinations.py
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Đảm bảo hiển thị UTF-8 trên Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from config import DATA_DIR, OOF_DIR, SUB_DIR, TARGET, ID
from utils import load_preds, save_preds
from blend import to_logit, load_y_folds

warnings.filterwarnings("ignore")
T0 = time.time()

def log(*args):
    print(f"[{time.time() - T0:5.1f}s]", *args, flush=True)

print("=" * 75)
print("     CS116 KAGGLE S6E10 - BỘ THỬ NGHIỆM 3 TỔ HỢP MÔ HÌNH BỨT PHÁ")
print("=" * 75)

# 1. Nạp dữ liệu nhãn và fold chuẩn chung
log("1. Nạp nhãn thật và 5-fold Stratified chung từ folds.csv...")
y_true, folds = load_y_folds()
N_TRAIN = len(y_true)

sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
N_TEST = len(sample)

# File tham chiếu P247 (LB 0.96183)
REF_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_score", "sub_track3_fw_p247_x_96183_w03.csv")
ref_test_preds = None
if os.path.exists(REF_CSV):
    ref_df = pd.read_csv(REF_CSV)
    ref_test_preds = ref_df["satisfaction"].values
    log(f"   ✓ Tìm thấy file tham chiếu P247: {REF_CSV} (LB 0.96183)")
else:
    log("   ! Không tìm thấy file tham chiếu P247 trong ref_score/ (bỏ qua bước blend P247)")

# 2. Danh mục các mô hình Baseline hiện tại của nhóm
BASE_MEMBERS = [
    ("quan_cat_dual_teacher_v4", "GBDT (CatBoost Dual Teacher)"),
    ("quan_xgb_dual_teacher_v4", "GBDT (XGBoost Dual Teacher)"),
    ("quan_xgb_safe_teacher_v4", "GBDT (XGBoost Safe Teacher)"),
    ("quan_xgb_original_aug_v4", "GBDT (XGBoost Original Aug)"),
    ("A_tabm_v1",                 "Deep Learning (TabM Batch Ensemble)"),
    ("cs116_llr_xgb_v1",         "GenAI (distilgpt2 LLR Surrogate)"),
]

# 2 mô hình mới được bạn xây dựng
NEW_MEMBERS = [
    ("B_personal_spec_v1", "Specialist GBDT (Personal Travel Dual-Branch)"),
    ("B_realmlp_v1",       "Deep Learning (RealMLP-TD Neural Net)"),
]

log("\n2. Kiểm tra các file OOF & Test trong thư mục oof/...")
loaded_models = {}

for name, desc in BASE_MEMBERS + NEW_MEMBERS:
    try:
        oof_p, test_p = load_preds(name)
        auc = roc_auc_score(y_true, oof_p)
        loaded_models[name] = {"oof": oof_p, "test": test_p, "desc": desc, "auc": auc}
        log(f"   [OK] {name:<26} ({desc[:35]}...): CV AUC = {auc:.6f}")
    except FileNotFoundError:
        log(f"   [MISSING] {name:<26}: Chưa có trong oof/ (Cần chạy notebook để sinh file này)")

# Kiểm tra điều kiện tối thiểu
base_available = [name for name, _ in BASE_MEMBERS if name in loaded_models]
assert len(base_available) >= 2, "Cần có ít nhất 2 model baseline trong oof/!"

# Hàm tiện ích huấn luyện Meta Stacker
def train_meta_stacker(member_names, c_value=0.003):
    """Huấn luyện Logistic Regression Stacker trên thang đo Logit qua 5 fold sạch."""
    Z_list = [to_logit(loaded_models[m]["oof"], loaded_models[m]["test"]) for m in member_names]
    X_train = np.column_stack([z[0] for z in Z_list])
    X_test = np.column_stack([z[1] for z in Z_list])
    
    oof_stack = np.zeros(N_TRAIN, dtype=np.float32)
    test_stack = np.zeros(N_TEST, dtype=np.float32)
    coefs = []
    
    for k in np.unique(folds):
        tr_mask = folds != k
        va_mask = folds == k
        
        clf = LogisticRegression(C=c_value, tol=1e-5, max_iter=3000)
        clf.fit(X_train[tr_mask], y_true[tr_mask])
        
        oof_stack[va_mask] = clf.decision_function(X_train[va_mask])
        test_stack += clf.decision_function(X_test) / len(np.unique(folds))
        coefs.append(clf.coef_[0])
        
    oof_prob = 1 / (1 + np.exp(-oof_stack))
    test_prob = 1 / (1 + np.exp(-test_stack))
    cv_auc = roc_auc_score(y_true, oof_prob)
    mean_coef = np.mean(coefs, axis=0)
    
    return oof_prob, test_prob, cv_auc, mean_coef

# 3. Đo lường Baseline Reference (Case 0)
log("\n" + "=" * 65)
log("--- ĐO LƯỜNG MỐC BASELINE: TEAM STACK V2 HIỆN TẠI ---")
oof_base, test_base, cv_base, coef_base = train_meta_stacker(base_available)
log(f"★ Baseline Team Stack ({len(base_available)} models) CV AUC = {cv_base:.6f} ★")

# Định nghĩa 3 trường hợp thử nghiệm
EXPERIMENT_CASES = [
    {
        "case_id": "Case 1",
        "title": "Tổ hợp 1: Baseline Team + Personal Travel Specialist (B_personal_spec_v1)",
        "models": base_available + (["B_personal_spec_v1"] if "B_personal_spec_v1" in loaded_models else []),
        "target_new": ["B_personal_spec_v1"],
        "tag": "case1_personal_spec"
    },
    {
        "case_id": "Case 2",
        "title": "Tổ hợp 2: Baseline Team + RealMLP Neural Network (B_realmlp_v1)",
        "models": base_available + (["B_realmlp_v1"] if "B_realmlp_v1" in loaded_models else []),
        "target_new": ["B_realmlp_v1"],
        "tag": "case2_realmlp"
    },
    {
        "case_id": "Case 3",
        "title": "Tổ hợp 3: Baseline Team + CẢ HAI (B_personal_spec_v1 + B_realmlp_v1)",
        "models": base_available + [m for m in ["B_personal_spec_v1", "B_realmlp_v1"] if m in loaded_models],
        "target_new": ["B_personal_spec_v1", "B_realmlp_v1"],
        "tag": "case3_both_models"
    }
]

SUMMARY_RESULTS = []

# Chạy thử nghiệm từng trường hợp
for exp in EXPERIMENT_CASES:
    log("\n" + "=" * 75)
    log(f"BẮT ĐẦU: {exp['case_id']} - {exp['title']}")
    log("=" * 75)
    
    # Kiểm tra xem các model mới đã có file chưa
    missing_in_case = [m for m in exp["target_new"] if m not in loaded_models]
    if missing_in_case:
        log(f"   [CHƯA SẴN SÀNG] Thiếu file OOF: {missing_in_case}")
        log(f"   -> Vui lòng chạy notebook tương ứng trong notebooks/ để tạo file OOF trước khi chạy tổ hợp này.")
        SUMMARY_RESULTS.append({
            "Case": exp["case_id"],
            "Mô hình thêm": ", ".join(exp["target_new"]),
            "Status": "Chưa có file OOF",
            "CV AUC": "-",
            "Delta CV": "-",
            "Weight": "-",
            "Tương quan P247": "-"
        })
        continue
        
    # Tính ma trận tương quan giữa các mô hình trong tổ hợp
    m_names = exp["models"]
    log(f"1. Phân tích tương quan đa dạng giữa {len(m_names)} mô hình...")
    R = np.column_stack([rankdata(loaded_models[m]["oof"]) / N_TRAIN for m in m_names])
    corr_mat = pd.DataFrame(np.corrcoef(R, rowvar=False), index=m_names, columns=m_names)
    print(corr_mat.round(4).to_string())
    
    # Huấn luyện Meta-Stacker
    log(f"\n2. Huấn luyện Meta Stacker Logistic Regression...")
    oof_case, test_case, cv_case, mean_coef = train_meta_stacker(m_names)
    delta_cv = cv_case - cv_base
    
    log(f"\n   ★ {exp['case_id']} CV AUC = {cv_case:.6f} (Delta CV: {delta_cv:+.6f}) ★")
    
    # Hiển thị trọng số
    order = np.argsort(-np.abs(mean_coef))
    log("   Trọng số đóng góp của từng mô hình (Weights):")
    weight_str_list = []
    for idx in order:
        m_name = m_names[idx]
        w = mean_coef[idx]
        marker = " << [MODEL MỚI]" if m_name in exp["target_new"] else ""
        print(f"      {m_name:<28} : {w:+.4f}{marker}")
        if m_name in exp["target_new"]:
            weight_str_list.append(f"{m_name}: {w:+.4f}")
            
    # Lưu OOF và Test của tổ hợp
    stack_name = f"stack_{exp['tag']}"
    save_preds(stack_name, oof_case, test_case, y_true)
    
    # Xuất file nộp bài độc lập thuần của tổ hợp
    sub_pure = pd.DataFrame({ID: sample[ID], TARGET: test_case})
    pure_sub_path = SUB_DIR / f"sub_{stack_name}_pure.csv"
    sub_pure.to_csv(pure_sub_path, index=False)
    log(f"   -> Đã lưu bài nộp thuần của tổ hợp: {pure_sub_path}")
    
    corr_p247_str = "-"
    if ref_test_preds is not None:
        # Tính tương quan với P247
        r_ref = rankdata(ref_test_preds) / (len(ref_test_preds) + 1)
        r_case = rankdata(test_case) / (len(test_case) + 1)
        corr_ref, _ = spearmanr(r_ref, r_case)
        corr_p247_str = f"{corr_ref:.4f}"
        log(f"   Tương quan rank với P247 Benchmark: {corr_ref:.4f}")
        
        # Tạo 3 mức nộp bài kết hợp với P247
        # 1. w05: 5% Team + 95% P247 (An toàn)
        b_w05 = 0.95 * r_ref + 0.05 * r_case
        sub_w05 = pd.DataFrame({ID: sample[ID], TARGET: rankdata(b_w05) / (len(b_w05) + 1)})
        out_w05 = SUB_DIR / f"sub_{exp['tag']}_blend_w05.csv"
        sub_w05.to_csv(out_w05, index=False)
        
        # 2. w10: 10% Team + 90% P247 (Mức cân bằng tối ưu)
        b_w10 = 0.90 * r_ref + 0.10 * r_case
        sub_w10 = pd.DataFrame({ID: sample[ID], TARGET: rankdata(b_w10) / (len(b_w10) + 1)})
        out_w10 = SUB_DIR / f"sub_{exp['tag']}_blend_w10.csv"
        sub_w10.to_csv(out_w10, index=False)
        
        # 3. w15: 15% Team + 85% P247 (Đột phá)
        b_w15 = 0.85 * r_ref + 0.15 * r_case
        sub_w15 = pd.DataFrame({ID: sample[ID], TARGET: rankdata(b_w15) / (len(b_w15) + 1)})
        out_w15 = SUB_DIR / f"sub_{exp['tag']}_blend_w15.csv"
        sub_w15.to_csv(out_w15, index=False)
        
        log(f"   ✓ Đã tạo 3 file blend P247:")
        log(f"     - w05 (An toàn):   {out_w05}")
        log(f"     - w10 (Khuyến nghị): {out_w10}")
        log(f"     - w15 (Đột phá):   {out_w15}")

    SUMMARY_RESULTS.append({
        "Case": exp["case_id"],
        "Mô hình thêm": ", ".join(exp["target_new"]),
        "Status": "Hoàn thành",
        "CV AUC": f"{cv_case:.6f}",
        "Delta CV": f"{delta_cv:+.6f}",
        "Weight": ", ".join(weight_str_list),
        "Tương quan P247": corr_p247_str
    })

# Báo cáo tổng kết so sánh
print("\n" + "=" * 85)
print("                    BẢNG TỔNG HỢP SO SÁNH 3 TỔ HỢP MÔ HÌNH")
print("=" * 85)
sum_df = pd.DataFrame(SUMMARY_RESULTS)
print(sum_df.to_string(index=False))
print("=" * 85)
print("HƯỚNG DẪN HÀNH ĐỘNG DÀNH CHO BẠN:")
print("1. Nếu Case 1 hoặc Case 2 hoặc Case 3 chưa có file OOF:")
print("   - Mở Jupyter Notebook: notebooks/B_personal_specialist_gbdt.ipynb và chạy toàn bộ để sinh B_personal_spec_v1.")
print("   - Mở Jupyter Notebook: notebooks/B_realmlp_pytabkit.ipynb và chạy toàn bộ để sinh B_realmlp_v1.")
print("2. Sau khi chạy xong 1 hoặc cả 2 notebook, chạy lại script này: python run_experiment_combinations.py")
print("3. Chọn file sub có mức w10 (10% Team + 90% P247) hoặc w15 của tổ hợp có CV AUC cao nhất để submit lên Kaggle!")
print("=" * 85)
