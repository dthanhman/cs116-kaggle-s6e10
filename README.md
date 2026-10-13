# CS116: Kaggle Playground S6E10 - Predicting Airline Satisfaction

Repo chung của nhóm cho cuộc thi [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10).

- **Bài toán:** dự đoán xác suất hành khách hài lòng (`satisfaction`).
- **Metric:** AUC, điểm từ 0,5 (đoán bừa) đến 1 (hoàn hảo). Chỉ thứ tự của xác suất là quan trọng, không cần chỉnh ngưỡng.
- **Mục tiêu:** top 5% leaderboard.
- **Hạn nộp:** 31/10/2026 23:59 UTC, tức **6:59 sáng 1/11** giờ Việt Nam. Kiểm tra lại trên trang Kaggle vì hạn có thể đổi.

Đọc hết file này một lần trước khi code. Gặp từ lạ thì tra [mục 0](#0-thuật-ngữ). Kế hoạch đầy đủ (lịch, phân vai, nhiệm vụ) nằm trong [docs/KE_HOACH.md](docs/KE_HOACH.md).

---

## 0. Thuật ngữ

| Từ | Nghĩa |
| --- | --- |
| **AUC** | Metric của cuộc thi, từ 0,5 (đoán bừa) đến 1 (hoàn hảo). Đo xem model có xếp người hài lòng lên trên người không hài lòng không, nên chỉ thứ tự của xác suất là quan trọng |
| **Fold** | Một phần của tập train. Ta chia train thành 5 fold; mỗi lần lấy 1 fold làm validation (để chấm điểm), 4 fold còn lại để train |
| **CV** (cross-validation) | Làm như trên 5 lần, mỗi fold làm validation một lần. **CV AUC** là điểm AUC tính trên toàn bộ train theo cách này. Đây là điểm chính để so sánh model |
| **OOF** (out-of-fold) | Dự đoán của model trên mỗi dòng train, lấy từ lần dòng đó nằm trong fold validation (model chưa thấy dòng đó khi train). Dùng để tính CV AUC và để blend |
| **LB** (leaderboard) | Bảng xếp hạng trên Kaggle. **Public LB** là điểm thấy được trong lúc thi, chỉ chấm trên một phần của test. **Private LB** chấm trên phần còn lại, công bố khi cuộc thi kết thúc và là điểm xếp hạng cuối cùng |
| **Seed** | Số cố định cho phần ngẫu nhiên (chia fold, khởi tạo model) để chạy lại ra cùng kết quả. Nhóm dùng seed 42 |
| **Baseline** | Model đơn giản đầu tiên để có điểm mốc, sau đó cải thiện dần |
| **EDA** | Khám phá dữ liệu: xem phân phối, tỉ lệ hài lòng theo từng cột, tương quan |
| **Adversarial validation** | Train một model phân biệt dòng train với dòng test. AUC gần 0,5 nghĩa là train và test giống nhau (tốt) |
| **GBDT** | Nhóm model cây tăng cường: **LightGBM** (viết tắt `lgb`), **XGBoost** (`xgb`), **CatBoost** (`cat`). Thường mạnh nhất với dữ liệu bảng |
| **MLP** | Mạng nơ-ron nhiều lớp đơn giản cho dữ liệu bảng |
| **Feature engineering** (`fe`) | Tạo cột mới từ cột có sẵn, ví dụ trung bình các điểm dịch vụ |
| **Tuning** | Dò tham số tốt nhất cho model. **Optuna** là thư viện tự động hóa việc này |
| **Blend** | Trộn dự đoán của nhiều model. **Rank-blend**: đổi xác suất của từng model thành thứ hạng rồi lấy trung bình |
| **Stacking** | Dùng một model nhỏ học cách kết hợp OOF của các model khác, thay cho lấy trung bình |
| **Dữ liệu gốc** | Bộ [binary-aviation-satisfaction-129k](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k) (khoảng 129 nghìn dòng thật) mà dữ liệu cuộc thi được sinh ra từ đó |

---

## 1. Cài đặt (làm một lần)

```bash
git clone https://github.com/dthanhman/cs116-kaggle-s6e10.git
cd cs116-kaggle-s6e10
pip install -r requirements.txt
```

1. Vào trang cuộc thi, bấm **Join Competition** và chấp nhận luật. Chưa làm bước này thì không tải được data.
2. Tải `train.csv`, `test.csv`, `sample_submission.csv` và bỏ vào thư mục `data/`.
3. Tải **`folds.csv`** từ thư mục `data/` trên [Drive chung của nhóm](https://drive.google.com/drive/folders/1V-WXI14-lKM9QfLIoJ048X4kE2XgJR-D?usp=sharing) và bỏ vào `data/`. **Không tự tạo fold.**
4. Mở `notebooks/00_template_model.ipynb`, chạy hết. Nếu in ra `CV AUC` khoảng 0,92 và lưu được `oof/oof_logreg.npy` thì máy bạn đã sẵn sàng.

Nếu chạy trên Kaggle Notebook hoặc Colab: đặt biến môi trường `DATA_DIR` (và `OOF_DIR` nếu cần) trỏ tới thư mục chứa data trước khi import `src/`.

---

## 2. Cấu trúc thư mục

```
data/            Dữ liệu (KHÔNG commit)
  train.csv, test.csv, sample_submission.csv   <- tải từ Kaggle
  folds.csv                                    <- A tạo, cả nhóm dùng chung
oof/             Dự đoán của từng model (KHÔNG commit, chia sẻ qua Drive)
  oof_<tên>.npy    dự đoán trên train, mỗi dòng do model chưa thấy dòng đó dự đoán (OOF)
  test_<tên>.npy   dự đoán trên test (trung bình 5 model của 5 fold)
submissions/     File CSV để nộp Kaggle (KHÔNG commit)
notebooks/       Notebook của từng người
  00_template_model.ipynb   <- mẫu, copy ra để làm model mới
src/             Code dùng chung, KHÔNG sửa tùy tiện (báo nhóm trước khi sửa)
  config.py      Đường dẫn, N_FOLDS=5, SEED=42, danh sách cột
  data.py        load_data(): đọc train/test + gắn cột fold, target đổi sang 0/1
  utils.py       save_preds() / load_preds(): lưu và đọc OOF, test preds đúng quy ước
  make_folds.py  Tạo folds.csv (chỉ A chạy, một lần)
  blend.py       Trộn dự đoán nhiều model theo thứ hạng (rank-blend) và in AUC
  submit.py      Tạo file nộp từ test_<tên>.npy và kiểm tra định dạng
experiments.csv  Nhật ký thí nghiệm: mỗi model một dòng
docs/KE_HOACH.md Kế hoạch đầy đủ: mục tiêu, lịch, phân vai, danh sách nhiệm vụ
CLAUDE.md        Cấu hình cho Claude Code (không cần đọc nếu không dùng Claude)
```

---

## 3. Sáu quy tắc bắt buộc

1. **Chỉ dùng `data/folds.csv` để chia dữ liệu.** File này ghi mỗi dòng train thuộc fold nào (0 đến 4); `load_data()` gắn sẵn cột `fold`. Nếu mỗi người tự chia một kiểu thì điểm CV của các model không so sánh được với nhau, và cũng không trộn (blend) dự đoán của nhau được.
2. **Mỗi model lưu đúng 2 file** qua `save_preds("<tên>", oof, test_pred, y)`:
   - `oof_<tên>.npy`: 699.635 giá trị, cùng thứ tự với `train.csv`. Mỗi giá trị là dự đoán cho một dòng train, lấy lúc dòng đó nằm trong fold dùng để chấm điểm (model chưa thấy nó). Dùng để tính CV AUC và để blend.
   - `test_<tên>.npy`: 299.844 giá trị, cùng thứ tự với `test.csv`. Dùng để nộp bài.

   Đặt tên rõ ràng, ví dụ `lgb_v1`, `cat_fe2`, `mlp_base`. **Không ghi đè** file của người khác.
3. **Nộp xác suất, không nộp nhãn 0/1.** Không cần chỉnh ngưỡng vì AUC chỉ quan tâm thứ hạng.
4. **Dữ liệu gốc (khoảng 129 nghìn dòng thật) chỉ được thêm vào phần dùng để train.** Ví dụ khi fold 0 dùng để chấm điểm, chỉ ghép dữ liệu gốc vào 4 fold còn lại; fold 0 vẫn chỉ có dữ liệu cuộc thi. Nếu không, điểm CV sẽ cao giả. Có dùng dữ liệu gốc hay không thì quyết bằng việc CV có tăng không.
5. **Tin điểm CV hơn điểm public LB.** Public LB chỉ chấm trên một phần của test nên dễ dao động, còn CV tính trên toàn bộ khoảng 700 nghìn dòng train nên ổn định hơn. Nếu một thay đổi làm CV tăng mà public LB giảm nhẹ, vẫn tin CV. Chạy theo public LB dễ bị tụt hạng khi private LB công bố. Nếu hai điểm lệch nhau nhiều, báo A để kiểm tra.
6. **Chỉ A nộp bài lên Kaggle**, để kiểm soát số lần nộp mỗi ngày. Ai muốn nộp thì gửi tên file `test_<tên>.npy` cho A.

---

## 4. Quy trình làm một model

1. Copy `notebooks/00_template_model.ipynb` thành `notebooks/<A|B|C>_<tên>.ipynb`, ví dụ `B_lgb_baseline.ipynb`.
2. Sửa 3 chỗ `TODO`: tên model, hàm `make_features`, hàm `fit_predict`.
3. Chạy hết notebook. Nó in AUC từng fold, CV AUC, rồi lưu file vào `oof/`.
4. Thêm một dòng vào `experiments.csv`:
   ```
   2026-10-15,B,lgb_v1,0.95123,,baseline 1000 cây learning rate 0.05
   ```
   Thứ tự cột: ngày, người làm, tên model, CV AUC, LB AUC, ghi chú. Cột LB AUC để trống, A điền sau khi nộp. Con số `0.95123` ở đây chỉ là ví dụ.
5. Upload `oof_<tên>.npy` và `test_<tên>.npy` lên thư mục `oof/` trên [Drive chung](https://drive.google.com/drive/folders/1V-WXI14-lKM9QfLIoJ048X4kE2XgJR-D?usp=sharing).
6. **Xóa output notebook** (*Clear All Outputs*), rồi commit và push.

Kiểm tra blend và tạo file nộp:

```bash
cd src
python blend.py lgb_v1 mlp_base              # in AUC từng model và AUC blend
python blend.py lgb_v1 mlp_base --save bl1   # lưu blend thành oof_bl1 / test_bl1
python submit.py bl1                         # -> submissions/sub_bl1.csv (A nộp)
```

---

## 5. Phân vai và việc trước mắt

| Người | Tên | Vai trò | Việc tuần này (12-16/10) |
| --- | --- | --- | --- |
| A | `CHƯA RÕ` | Dữ liệu, cách chấm điểm, quản lý repo và nộp bài | Tạo `folds.csv`, repo, `submit.py` (xong); khám phá dữ liệu (EDA) và kiểm tra train với test có giống nhau không (adversarial validation); nộp model đầu tiên 16/10 |
| B | `CHƯA RÕ` | Model cây (GBDT: LightGBM, XGBoost, CatBoost) và tạo feature mới | Model LightGBM đầu tiên → `oof_lgb.npy` trước 16/10 |
| C | `CHƯA RÕ` | Mạng nơ-ron (MLP) và trộn các model (ensemble) | Model MLP đầu tiên → `oof_mlp.npy`; hoàn thiện `blend.py` trước 16/10 |

Lịch tổng thể:

| Giai đoạn | Ngày | Nội dung |
| --- | --- | --- |
| Khởi động | 12-13/10 | Data, `folds.csv`, repo |
| Baseline | 13-16/10 | Model đầu tiên (LightGBM, MLP), EDA, nộp bài đầu |
| Cải thiện | 17-27/10 | Tạo feature mới, thêm XGBoost/CatBoost, dò tham số (tuning), thử dữ liệu gốc, thêm model khác loại |
| Ensemble | 25-28/10 | Trộn dự đoán các model (blend/stacking), chạy lại model tốt nhất với nhiều seed |
| **Đóng băng** | **29/10** | Không đổi model hay feature nữa, chốt 2 bài cuối |
| Dự phòng | 30-31/10 | Chỉ nộp và xác nhận |

Danh sách 21 nhiệm vụ đầy đủ nằm ở mục 5 của [docs/KE_HOACH.md](docs/KE_HOACH.md). Họp 15 phút mỗi 2-3 ngày.

---

## 6. Thông tin dữ liệu (đã kiểm tra 12/10)

- `train.csv` có 699.635 dòng, `test.csv` có 299.844 dòng. `id` của train từ 0 đến 699634, của test từ 699635 đến 999478.
- Target `satisfaction` có kiểu **bool**: True 44,36%, False 55,64%. `load_data()` đã đổi sang 0/1.
- Giá trị thiếu chỉ có ở `Arrival Delay in Minutes`: 292 dòng train, 130 dòng test.
- 21 feature:
  - **4 category dạng chữ:** `Gender`, `Customer Type`, `Type of Travel`, `Class` (Business / Eco / Eco Plus).
  - **13 điểm dịch vụ 0-5:** điểm 0 thường là "không áp dụng"; `Baggage handling` không có điểm 0.
  - **4 cột số:** `Age`, `Flight Distance`, `Departure Delay in Minutes`, `Arrival Delay in Minutes`.
- Hầu hết giá trị delay bằng 0 (trung vị bằng 0). Cần lưu ý khi ghép với dữ liệu gốc.
- Không có dòng trùng, `id` không liên quan đến target, phân phối train và test gần như giống nhau.

---

## 7. Quy ước Git

- Mỗi người chỉ sửa notebook của mình. Muốn sửa `src/` thì báo nhóm trước.
- `git pull` trước khi bắt đầu làm, commit nhỏ và thường xuyên, message ngắn gọn (ví dụ `B: lgb baseline cv 0.951`).
- Không commit `data/`, `oof/`, `submissions/`, file `.npy` hay `.csv` lớn. `.gitignore` đã chặn sẵn những thứ này.
