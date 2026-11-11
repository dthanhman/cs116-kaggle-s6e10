# CS116 Kaggle S6E10 - Báo Cáo Tổng Hợp Thử Nghiệm & Kết Quả Đột Phá

> **Thời gian thực hiện:** 18/10/2026  
> **Tác giả:** LuKimLuan-UIT (`24521028@gm.uit.edu.vn`)  
> **Cuộc thi:** [Kaggle Playground Series S6E10: Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10)  
> **Nhánh Git:** `feat/lukimluan-models` (Commit: `bc85e46`)  
> **Mục tiêu:** Phá vỡ nút thắt điểm số `0.96184` và tiến vào Top 5% Leaderboard.

---

## 1. Bối cảnh & Phân tích Nút Thắt (Bottleneck Diagnosis)

### Hiện tượng:
* Ban đầu, file nộp `sub_final_p247_team_blend_w05.csv` đạt Public LB **`0.96184`** (chỉ tăng đúng `+0.00001` so với mốc tham chiếu P247 `0.96183`).
* Khi thử nghiệm tăng tỷ trọng đóng góp của nhóm lên `w10` (10% Team + 90% P247), điểm số bị sụt giảm xuống **`0.96181`**; ở mức `w15`, điểm số tiếp tục tụt sâu hơn.

### Bản chất toán học:
1. **Pha loãng Ensemble (Ensemble Dilution):**
   * File tham chiếu P247 (`0.96183`) được tổng hợp từ **hơn 200 mô hình công khai** của các Grandmaster (Chris Deotte, Abdullah, Sadam...) với CV thực tế lên tới **`0.96220+`**.
   * Bộ Stack nội bộ ban đầu của nhóm chỉ gồm 6 mô hình cục bộ với CV đạt **`0.961175`**.
   * Việc cộng gộp tuyến tính một mô hình yếu hơn (CV `0.9612`) với tỷ trọng lớn (10% - 15%) vào một mô hình mạnh hơn (CV `0.9622`) sẽ làm suy giảm độ chính xác và xáo trộn thứ tự xếp hạng của toàn bộ test set.
2. **"Tử huyệt" trong dữ liệu cuộc thi:**
   * **Business Travel** (71% dữ liệu, ~497k dòng): Tỷ lệ hài lòng là **58.4%**, CV AUC đạt tới **`0.95764`**.
   * **Personal Travel** (29% dữ liệu, ~202k dòng): Tỷ lệ hài lòng chỉ **9.7%**, nhưng CV AUC tụt thảm hại xuống **`0.83362`** do 71% khách Business chi phối toàn bộ hàm mất mát của các mô hình cây thông thường.

---

## 2. Các Cải Tiến & Mô Hình Mới Đã Xây Dựng

Để giải quyết tình trạng bão hòa cây GBDT và khắc phục điểm yếu phân khúc, 2 mô hình độc lập và 1 pipeline thử nghiệm tự động đã được triển khai:

### 1. `B_personal_spec_v1` (Dual-Specialist GBDT)
* **File:** [`notebooks/B_personal_specialist_gbdt.ipynb`](../notebooks/B_personal_specialist_gbdt.ipynb)
* **Kiến trúc:** Huấn luyện 2 chuyên gia LightGBM độc lập:
  * `Model_Personal`: Tối ưu riêng trên các dòng Personal Travel với `scale_pos_weight = 1.35` (bù đắp nhãn dương hiếm 9.7%), phạt L2 cao (`reg_lambda = 20.0`), kết hợp đặc trưng độ trễ tương tác (`Severe_Delay`, `Delay_Per_Distance`) và dịch vụ mặt đất (`Ground_Hygiene_Score`).
  * `Model_Business`: Tối ưu riêng cho khách thương gia tập trung vào tiện ích cao cấp (Wifi, Inflight Entertainment, Leg room).
* **Kết quả đơn lẻ:** CV AUC đạt **`0.958643`**.

### 2. `B_realmlp_v1` (RealMLP-TD Deep Learning)
* **File:** [`notebooks/B_realmlp_pytabkit.ipynb`](../notebooks/B_realmlp_pytabkit.ipynb)
* **Kiến trúc:** Mạng nơ-ron sâu cho dữ liệu bảng từ thư viện `pytabkit`, tối ưu hóa trực tiếp trên GPU NVIDIA RTX 4050.
* **Tối ưu tốc độ & Bộ nhớ:**
  * Kích hoạt Tensor Cores: `torch.set_float32_matmul_precision('high')`.
  * Đặt `batch_size = 2048`, `n_epochs = 16` (thay vì mặc định 256 epoch): rút ngắn thời gian từ nhiều giờ xuống chỉ còn **~60 giây/fold**.
  * Đo đạc thực tế: VRAM chỉ tiêu thụ **`55.17 MB`** (dưới 1% dung lượng GPU).
* **Kết quả đơn lẻ:** CV AUC đạt **`0.957695`**.

### 3. Pipeline Đánh Giá & Stacking Đa Kịch Bản
* **File:** [`run_experiment_combinations.py`](../run_experiment_combinations.py)
* Tự động nạp toàn bộ OOF hiện có của nhóm, đối chiếu 3 trường hợp kết hợp (Case 1: Specialist, Case 2: RealMLP, Case 3: Cả hai) qua 5-fold CV sạch trên `folds.csv` và xuất bài nộp theo các mức tỷ trọng.

---

## 3. Bảng Kết Quả Thực Nghiệm Toàn Diện

### A. Ma trận Tương quan Đa dạng (Diversity Matrix - Spearman $\rho$)
| Mô hình đối chiếu | `quan_cat_dual_teacher_v4` | `quan_xgb_dual_teacher_v4` | `A_tabm_v1` | Đánh giá |
| :--- | :---: | :---: | :---: | :--- |
| **`B_personal_spec_v1`** | **0.9390** | 0.9382 | 0.9254 | Không gian lỗi rất khác biệt so với cây thường |
| **`B_realmlp_v1`** | **0.9297** | **0.9268** | **0.9017** | **Cực kỳ trực giao** (tương quan $< 0.93$) |

*(Ghi chú: Hai mô hình cây XGBoost và CatBoost thông thường của nhóm có tương quan với nhau lên tới $\rho > 0.995$).*

### B. Hiệu Quả Meta-Stacking 3 Kịch Bản (5-Fold Logistic Regression)
| Kịch bản | Danh sách mô hình | CV AUC | $\Delta \text{CV}$ so với Base | Trọng số mô hình mới (Meta-Weights) |
| :--- | :--- | :---: | :---: | :--- |
| **Baseline cũ** | 6 models (4 GBDT + TabM + LLR) | 0.961175 | 0.000000 | - |
| **Case 1** | Baseline + `B_personal_spec_v1` | 0.961207 | **+0.000032** | `B_personal_spec_v1`: **+0.1023** |
| **Case 2** | Baseline + `B_realmlp_v1` | 0.961182 | +0.000007 | `B_realmlp_v1`: **+0.0975** |
| **Case 3** | Baseline + **CẢ HAI MÔ HÌNH** | **0.961206** | **+0.000031** | `B_personal_spec_v1`: **+0.0813**<br>`B_realmlp_v1`: **+0.0703** |

*Cả 2 mô hình mới đều nhận trọng số dương lớn và ổn định trong mô hình meta, khẳng định giá trị đóng góp thông tin thực chất.*

### C. Đối Chiếu Kết Quả Nộp Bài Kaggle (Public Leaderboard)
| Bài nộp | Tỷ trọng kết hợp | Public LB | Nhận xét thực nghiệm |
| :--- | :--- | :---: | :--- |
| `sub_track3_fw_p247_x_96183_w03.csv` | 100% P247 Benchmark | **0.96183** | Mốc tham chiếu gốc từ 247 model công khai. |
| `sub_final_p247_team_blend_w05.csv` | 95% P247 + 5% Team Stack cũ | **0.96184** | Nhích nhẹ (+0.00001) do mức 5% giữ nguyên 99.99% khung thứ hạng P247. |
| `sub_case3_both_models_blend_w10.csv` | 90% P247 + 10% Team Case 3 | **0.96181** | Bị tụt điểm (-0.00003): chứng minh blend ngoài $> 5\%$ sẽ pha loãng tín hiệu mạnh của P247. |

---

## 4. Quản Lý Phiên Bản Git & Đóng Góp Nhóm

* **Repository:** `https://github.com/dthanhman/cs116-kaggle-s6e10.git`
* **Nhánh phát triển:** `feat/lukimluan-models` (tạo riêng, không ảnh hưởng nhánh `main`).
* **Tài khoản commit:** `LuKimLuan-UIT <24521028@gm.uit.edu.vn>`.
* **Kiểm soát rác & File nặng:**
  * Đã cấu hình [.gitignore](../.gitignore) loại bỏ toàn bộ file cache, backup (`*.bak`), file dự đoán lớn (`ref_score/`, `results*/`, `*.npy`, `*.zip`).
  * Chỉ đẩy mã nguồn sạch và tài liệu markdown, đảm bảo dung lượng commit nhẹ và an toàn.
* **Pull Request:** Đã sẵn sàng mở tại:  
  `https://github.com/dthanhman/cs116-kaggle-s6e10/pull/new/feat/lukimluan-models`

---

## 5. Đề Xuất Chiến Lược Tiếp Theo Để Bứt Phá Thực Sự

Qua chuỗi thí nghiệm trên, nhóm đã xác định được giới hạn của phương pháp cộng gộp tuyến tính cục bộ. Để thực sự vượt qua mốc `0.96184`, nhóm nên tập trung vào 2 chiến lược sau:

1. **Đưa OOF độc quyền lên Kaggle Dataset để chạy `cs116_stack_v7.py` (Ưu tiên số 1):**
   * Tải `oof_B_realmlp_v1.npy`, `oof_B_personal_spec_v1.npy`, `oof_quan_*`, `oof_A_tabm_v1` lên Kaggle.
   * Chạy pipeline Stacking v7 hoàn chỉnh: đưa các model độc quyền vào chung ma trận 247+ OOF công khai (Meta L2 Stacker tự điều chỉnh trọng số), kết hợp **3 LightGBM + 1 XGBoost GPU Residual Boosting**.
   * Đây là công thức duy nhất có khả năng đẩy CV lên **`0.96220+`** và LB lên **`0.96190+`**.
2. **Khai thác mốc tham chiếu công khai cao hơn (`0.96188`):**
   * Thay vì dùng file tham chiếu cũ `0.96183`, tải file dự đoán của giải pháp công khai `0.96188` (ewtherka) để làm mốc nền tảng mới.
