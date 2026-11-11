# CS116 Kaggle S6E10: Phân Tích Công Việc Nhóm & Đề Xuất Chiến Lược Tối Ưu Điểm Số

> **Ngày lập:** 18/10/2026  
> **Thư mục phân tích:** `notebooks/teammate_kaggle_notebook/`  
> **Mục tiêu:** Đạt Top 5% Leaderboard cuộc thi [Playground Series S6E10: Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10) (ngưỡng tham chiếu ~0.9618+).

---

## 1. Tổng quan các công việc teammate đã thực hiện

Trong thư mục `notebooks/teammate_kaggle_notebook/`, đồng đội của bạn đã xây dựng một hệ thống thi đấu Kaggle ở trình độ rất cao, chia thành 3 trụ cột chính:
1. **Mô hình riêng & Feature Engineering nâng cao của team (`cs116.ipynb`)**
2. **Các mô hình đa dạng hóa phi truyền thống (`cs116_tabm.py`, `cs116_llr_score.py`)**
3. **Kiến trúc Stacking quy mô lớn P247 & Sửa phần dư Residual (`cs116-stack.ipynb`, `cs116_stack_v4/v5/v7.py`)**

```
                          ┌────────────────────────────────────────────────────────┐
                          │          KHO 247+ OOF CÔNG KHAI (P247 POOL)            │
                          │   (135 Base + Abdullah TabPFN + Golem + Sadam + ...)   │
                          └──────────────────────────┬─────────────────────────────┘
                                                     │
   ┌────────────────────────────────┐                │
   │      OOF RIÊNG CỦA NHÓM        │                │
   │  - 4 XGB/CatBoost (cs116.ipynb)├────────────────┤ (Lọc OOF, kiểm tra drift,
   │  - TabM ICLR 2025 (cs116_tabm) │                │  loại bỏ trùng lặp >0.9999)
   │  - GenAI LLR (cs116_llr_score) │                │
   └────────────────────────────────┘                ▼
                                      ┌──────────────────────────────┐
                                      │       STACK V7 PIPELINE      │
                                      │  - Family-wise L2 Logistic   │
                                      │  - 3 LGB + 1 XGB Residual    │
                                      │  - Nested CV (3 Meta Seeds)  │
                                      └──────────────┬───────────────┘
                                                     │
                                                     ▼
                                      ┌──────────────────────────────┐
                                      │     SUBMISSION (TOP 5%)      │
                                      └──────────────────────────────┘
```

---

### Chi tiết từng thành phần

### Trụ cột 1: Huấn luyện mô hình riêng của nhóm (`cs116.ipynb`)
Tác giả: bạn Quân (`quan14333`). Notebook được thiết kế để chạy trên Kaggle GPU T4 với phiên bản `S6E10_Power_v4`:
* **Kỹ thuật Feature Engineering:**
  * **Contextual Target Encoding:** Tạo tương tác kết hợp phân khúc hành khách (`Type of Travel | Customer Type | Class`) ghép với 10 điểm dịch vụ và các cặp khoảng cách/độ tuổi (16 cột context).
  * **Route Profiling (Leave-One-Out):** Khai thác cột `Flight Distance` như một mã tuyến bay: tính số lượng chuyến bay cùng khoảng cách, trung bình điểm đánh giá, độ lệch chuẩn LOO (leave-one-out variance), tỷ lệ đánh giá cao ($\ge 4$), và tỷ lệ hành khách cùng phân khúc.
* **Khai thác dữ liệu gốc 129k (Original Data):**
  * Lọc trùng lặp nghiêm ngặt với tập cuộc thi.
  * Xây dựng **Safe Teacher**: Huấn luyện XGBoost và CatBoost chỉ trên dữ liệu gốc, lấy xác suất dự đoán làm feature bổ sung cho dữ liệu thi (hoàn toàn không gây leak).
  * **Original-Row Augmentation:** Ghép các dòng dữ liệu gốc vào 4 fold train với trọng số phạt `ORIG_WEIGHT = 0.35` (không đưa vào fold validation).
* **4 mô hình candidate sạch (OOF Clean):**
  * `quan_xgb_safe_teacher_v4`
  * `quan_xgb_dual_teacher_v4`
  * `quan_cat_dual_teacher_v4`
  * `quan_xgb_original_aug_v4`
  * Tuân thủ quy tắc chống rò rỉ: Trích 10% từ tập train nội bộ của từng fold làm tập dừng sớm (early stopping), sau đó refit mô hình trên 100% fold train trước khi dự đoán OOF validation và test.

---

### Trụ cột 2: Các mô hình đa dạng hóa (Diversity Models)

Trong Stacking, 10 mô hình cây giống nhau không bằng 1 mô hình thuộc họ khác biệt. Đồng đội đã xây dựng 2 kỹ thuật vô cùng độc đáo:

#### A. Deep Learning cho Tabular: TabM (`cs116_tabm.py`)
* Sử dụng kiến trúc **TabM** (Gorishniy et al., ICLR 2025): Mạng nơ-ron nhiều lớp (MLP) với cơ chế **Batch Ensemble $k$ đầu** và mã hóa số rời rạc theo đoạn **Piecewise Linear Embeddings** (`rtdl_num_embeddings`).
* Feature đầu vào: Dữ liệu đã qua `QuantileTransformer` (phân phối chuẩn), kết hợp logit của teacher LightGBM train từ dữ liệu gốc, và target encoding sạch trong fold.
* **Giá trị cho Stacking:** TabM là mạng nơ-ron (Deep Learning), tạo ra dự đoán có tương quan thấp hơn so với họ cây (GBDT), giúp mô hình meta-stacking bù đắp được những ca mà cây dự đoán sai.

#### B. GenAI LLR Surrogate (`cs116_llr_score.py`)
* Ý tưởng từ giải pháp Top 2/3 Kaggle S6E9: Dữ liệu S6E10 được sinh ra bằng mô hình ngôn ngữ (GPT-2 tokenizer).
* Cách thực hiện:
  * Fine-tune mô hình `distilgpt2` **chỉ trên tập dữ liệu gốc 129k** với định dạng văn bản hóa bảng kiểu GReaT (ví dụ: `"Class is Business, Flight Distance is 821, satisfaction is True"`).
  * Tính tỷ số log-likelihood (LLR):
    $$\text{LLR} = \log p(\text{features} \mid \text{satisfaction is True}) - \log p(\text{features} \mid \text{satisfaction is False})$$
* **Giá trị cho Stacking:** Vì mô hình chỉ học trên dữ liệu gốc, LLR là một feature **hoàn toàn label-free** trên dữ liệu cuộc thi nhưng mang thông tin cực mạnh về phân phối của bộ sinh dữ liệu.

---

### Trụ cột 3: Hệ thống Stacking P247 & Sửa phần dư (`cs116-stack.ipynb`, `cs116_stack_v4/v5/v7.py`)

Đây là trung tâm chỉ huy để gom toàn bộ kết quả tạo ra bài nộp cuối cùng:
1. **Kho OOF công khai P247:**
   * Nạp hơn 247 mô hình từ các Kaggler hàng đầu (Chris Deotte, Abdullah, Sadam, Kratosyan, Golem, Koumeimaki...).
   * **Cơ chế lọc tự động (Data Audit):**
     * Loại bỏ các OOF có AUC $< 0.90$ (lệch dòng) hoặc $> 0.9625$ (nghi là bản stack fit in-sample gây leak).
     * Loại bỏ mô hình bị drift giữa train và test ($> 0.03$).
     * Loại bỏ các mô hình gần như sao chép nhau (rank correlation $\rho > 0.9999$).
2. **Stack v4 & v5:**
   * Phân chia các mô hình theo họ (family): `gbdt`, `tabpfn`, `tfm` (Foundation Models), `mlp`.
   * **Family-wise L2 Regularized Logistic Regression:** Điều chỉnh độ co L2 riêng cho từng họ (ví dụ ưu tiên các họ TabPFN/TFM với trọng số gấp 1.5 - 3.0 lần).
   * Blend tỉ lệ: $0.8 \times \text{LR} + 0.2 \times \text{LightGBM}$.
3. **Stack v7 (Đỉnh cao kỹ thuật):**
   * Tái tạo công thức sạch của giải pháp đạt 0.96183 LB (Den Pugovkin) nhưng được đưa vào **Nested Cross-Validation** (3 meta-seeds: 42, 7, 101) để triệt tiêu may rủi.
   * **Cơ chế sửa phần dư (Residual Correction):**
     * Bước 1: Tính margin cơ sở $m = \text{decision\_function}(\text{LR})$.
     * Bước 2: Huấn luyện 3 cây LightGBM (seed 2026, 2027, 2028) và 1 XGBoost GPU để học riêng phần dư sai số (residual) trên $m$.
     * Bước 3: Cắt ngưỡng (clipping) phần sửa sai trong $[-0.5, 0.5]$ và cộng vào margin ban đầu:
       $$p = \sigma\left(m + \text{clip}(0.75 \cdot \overline{\text{LGB}}_{\text{res}} + 0.25 \cdot \text{XGB}_{\text{res}}, -0.5, 0.5)\right)$$
   * Đánh giá lưới cấu hình (`v7_grid.csv`) và chọn ra cấu hình vượt trội trên cả 3 meta seed và hơn 10/15 fold.

---

## 2. Đánh giá chung

* **Điểm mạnh:**
  * Kiến trúc thi đấu cực kỳ chặt chẽ, tư duy Stacking chuẩn xác theo đúng phương pháp của các Kaggle Grandmaster.
  * Tận dụng tối đa tài nguyên công khai (pool P247) kết hợp với các nghiên cứu riêng biệt (Route LOO, Safe Teacher, TabM, GPT-2 LLR).
  * Quy trình kiểm tra OOF và Nested CV 3 seed giúp loại bỏ hoàn toàn nguy cơ bị tụt hạng khi Private Leaderboard công bố (tránh hiện tượng "overfit Public LB").
* **Điểm nghẽn cần hoàn thiện ngay:**
  * Các mô hình trong `cs116_tabm.py` và `cs116_llr_score.py` chưa được xuất ra các file OOF npy chính thức để cắm vào pipeline Stack v7 (`load_team()`).
  * Cần đồng bộ toàn bộ OOF của nhóm lên Kaggle Dataset để chạy Stacking hoàn chỉnh.

---

## 3. Các đề xuất cải tiến để đạt điểm cao nhất (Proposals)

### Đề xuất 1: Hoàn tất xuất OOF cho TabM và LLR để nạp vào Pool của Nhóm (Ưu tiên số 1)
* **Lý do:** Pool 247 công khai đã có đến ~160 mô hình cây (LightGBM/XGBoost). Nếu ta chỉ đưa thêm XGBoost của ta vào, Stacker sẽ khó tăng điểm nhiều. Nhưng **TabM (Neural Network)** và **LLR (GPT-2)** là hai mô hình có bản chất toán học khác biệt 100%.
* **Hành động:**
  1. Chạy hoàn chỉnh 5 fold của `cs116_tabm.py` trên Kaggle GPU để xuất `oof_team_tabm_v1.npy` và `test_team_tabm_v1.npy`.
  2. Tạo 1 mô hình GBDT hoặc Ridge/LogReg sử dụng LLR làm feature chính để sinh ra `oof_team_llr_v1.npy`.
  3. Đưa vào thư mục OOF của nhóm và nạp vào `cs116_stack_v7.py` qua `@source("team", "cs116-s6e10-oof")`.

### Đề xuất 2: Phân khúc hóa Stacking (Segment-specific Meta-stacking)
* **Hiện tượng:** Qua phân tích EDA và notebook `cs116-stack.ipynb` (mục 4E), độ khó và AUC giữa hai nhóm khách hàng rất chênh lệch:
  * Khách **Business Travel**: AUC mô hình đạt tới ~`0.955+`.
  * Khách **Personal Travel**: AUC chỉ quanh mức ~`0.915`.
* **Đề xuất:**
  * Thay vì dùng 1 mô hình Logistic Regression chung cho toàn bộ 700k dòng, ta thử nghiệm **Dual-Segment Meta Model**:
    * Stacker $M_{\text{business}}$: Chỉ học trên các dòng khách Business.
    * Stacker $M_{\text{personal}}$: Chỉ học trên các dòng khách Personal.
  * Hoặc trong ma trận meta-features $Z$, bổ sung thêm cột tương tác: $Z \times \mathbb{I}(\text{Type of Travel} == \text{Business})$.

### Đề xuất 3: Tối ưu trọng số Ensemble bằng Nelder-Mead / Optuna
* **Hiện tại:** Trong `cs116_stack_v7.py`, các hệ số đang chọn theo lưới thô:
  * Blend nền: `0.8 LR + 0.2 LGB`.
  * Phần sửa: `0.75 LGB + 0.25 XGB`, clip `[-0.5, 0.5]`.
* **Đề xuất:**
  * Dùng thuật toán tối ưu không đạo hàm `scipy.optimize.minimize(method='Nelder-Mead')` hoặc Optuna để tìm vector trọng số $(w_1, w_2, w_3, w_4)$ tối ưu trực tiếp trên ROC AUC của OOF qua 5 fold.

### Đề xuất 4: Chiến lược chọn 2 bài nộp cuối cùng (Final Submissions - 29/10)
Theo quy định của Kaggle, nhóm được chọn 2 bài nộp cuối cùng để tính Private LB:
1. **Bài 1 (Best CV Candidate):**
   * Bản **Stack v7** tốt nhất sau khi đã tích hợp đầy đủ OOF của team (XGBoost + CatBoost + TabM + P247).
   * Lấy trung bình dự đoán qua 3 meta seeds (42, 7, 101) để làm mượt xác suất.
2. **Bài 2 (Robust Conservative Blend):**
   * Bản Rank-Average giữa **Stack v7** và **Stack v5** (hoặc v4):
     $$\text{Final} = 0.6 \cdot \text{Rank}(\text{Stack v7}) + 0.4 \cdot \text{Rank}(\text{Stack v5})$$
   * Bài này đóng vai trò bảo hiểm: nếu phần sửa sai của cây (residual) có chút overfit trên Public test thì bản Rank-Average bảo toàn thứ hạng sẽ giữ vững vị trí trong Top 5% Private LB.

---

## 4. Kế hoạch hành động gợi ý cho bạn (Action Plan)

| Bước | Việc cần làm | Công cụ / File thực hiện |
|---|---|---|
| **1** | Kiểm tra tiến độ chạy của Quân trên Kaggle, tải các file `oof_*.npy` về thư mục `oof/` | Kaggle Notebook `cs116.ipynb` |
| **2** | Chạy hoặc kiểm tra kết quả của `cs116_tabm.py` để lấy cặp OOF Deep Learning | Kaggle GPU T4 (`tabm`) |
| **3** | Cập nhật bộ dataset OOF của nhóm trên Kaggle (gồm các file npy mới) | Kaggle Dataset `cs116-s6e10-oof` |
| **4** | Chạy `cs116_stack_v7.py` trên Kaggle có gắn dataset OOF nhóm để sinh file submission cuối cùng | Kaggle GPU / `cs116_stack_v7.py` |
| **5** | Đối chiếu điểm CV và nộp thử bản Stack v7 đầu tiên lên Kaggle để lấy điểm Public LB mốc | `sub_stack_v7_best.csv` |
