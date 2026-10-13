# CS116: Kaggle Playground S6E10 - Predicting Airline Satisfaction

Repo chung của nhóm cho cuộc thi [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10).

- **Bài toán:** dự đoán xác suất hành khách hài lòng (`satisfaction`).
- **Metric:** AUC. Chỉ thứ hạng của xác suất là quan trọng.
- **Mục tiêu:** top 5% leaderboard.
- **Hạn nộp:** 31/10/2026 23:59 UTC, tức **6:59 sáng 1/11** giờ Việt Nam. Kiểm tra lại trên trang Kaggle vì hạn có thể đổi.

Đọc hết file này một lần trước khi code. Kế hoạch đầy đủ (lịch, phân vai, nhiệm vụ) nằm trong [docs/KE_HOACH.md](docs/KE_HOACH.md).

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
4. Mở `notebooks/00_template_model.ipynb`, chạy hết. Nếu in ra `CV AUC` và lưu được `oof/oof_logreg.npy` thì máy bạn đã sẵn sàng.

Nếu chạy trên Kaggle Notebook hoặc Colab: đặt biến môi trường `DATA_DIR` (và `OOF_DIR` nếu cần) trỏ tới thư mục chứa data trước khi import `src/`.

---

## 2. Cấu trúc thư mục

```
data/            Dữ liệu (KHÔNG commit)
  train.csv, test.csv, sample_submission.csv   <- tải từ Kaggle
  folds.csv                                    <- A tạo, cả nhóm dùng chung
oof/             Dự đoán của từng model (KHÔNG commit, chia sẻ qua Drive)
  oof_<tên>.npy    dự đoán trên train (out-of-fold)
  test_<tên>.npy   dự đoán trên test
submissions/     File CSV để nộp Kaggle (KHÔNG commit)
notebooks/       Notebook của từng người
  00_template_model.ipynb   <- mẫu, copy ra để làm model mới
src/             Code dùng chung, KHÔNG sửa tùy tiện (báo nhóm trước khi sửa)
  config.py      Đường dẫn, N_FOLDS=5, SEED=42, danh sách cột
  data.py        load_data(): đọc train/test + gắn cột fold, target đổi sang 0/1
  utils.py       save_preds() / load_preds(): lưu và đọc OOF, test preds đúng quy ước
  make_folds.py  Tạo folds.csv (chỉ A chạy, một lần)
  blend.py       Rank-blend nhiều model và in AUC
  submit.py      Tạo file nộp từ test_<tên>.npy và kiểm tra định dạng
experiments.csv  Nhật ký thí nghiệm: mỗi model một dòng
docs/KE_HOACH.md Kế hoạch đầy đủ: mục tiêu, lịch, phân vai, danh sách nhiệm vụ
CLAUDE.md        Cấu hình cho Claude Code (không cần đọc nếu không dùng Claude)
```

---

## 3. Năm quy tắc bắt buộc

1. **Chỉ dùng `data/folds.csv`.** Mọi model chia train/validation theo cột `fold` (gọi `load_data()` là có sẵn). Nếu mỗi người chia fold một kiểu thì không blend được và CV mất ý nghĩa.
2. **Mỗi model lưu đúng 2 file** qua `save_preds("<tên>", oof, test_pred, y)`:
   - `oof_<tên>.npy`: 699.635 giá trị, cùng thứ tự với `train.csv`
   - `test_<tên>.npy`: 299.844 giá trị, cùng thứ tự với `test.csv`

   Đặt tên rõ ràng, ví dụ `lgb_v1`, `cat_fe2`, `mlp_base`. **Không ghi đè** file của người khác.
3. **Nộp xác suất, không nộp nhãn 0/1.** Không cần chỉnh ngưỡng vì AUC chỉ quan tâm thứ hạng.
4. **Dữ liệu gốc 129k dòng chỉ được thêm vào phần train của mỗi fold**, không bao giờ đưa vào phần validation. Có dùng hay không thì quyết bằng CV.
5. **Tin CV hơn LB công khai.** Chỉ **A** nộp bài lên Kaggle để kiểm soát số lần submit. Ai muốn nộp thì gửi tên file `test_<tên>.npy` cho A.

---

## 4. Quy trình làm một model

1. Copy `notebooks/00_template_model.ipynb` thành `notebooks/<A|B|C>_<tên>.ipynb`, ví dụ `B_lgb_baseline.ipynb`.
2. Sửa 3 chỗ `TODO`: tên model, hàm `make_features`, hàm `fit_predict`.
3. Chạy hết notebook. Nó in AUC từng fold, CV AUC, rồi lưu file vào `oof/`.
4. Thêm một dòng vào `experiments.csv`:
   ```
   2026-10-15,B,lgb_v1,0.95123,,baseline 1000 cây lr 0.05
   ```
   Cột `lb_auc` để trống, A điền sau khi nộp.
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
| A | `CHƯA RÕ` | Data & Validation, Tech Lead | Tạo `folds.csv`, repo, `submit.py` (xong); EDA, adversarial validation; nộp baseline đầu 16/10 |
| B | `CHƯA RÕ` | GBDT & Feature Engineering | Baseline LightGBM → `oof_lgb.npy` trước 16/10 |
| C | `CHƯA RÕ` | Neural Net & Ensemble | Baseline MLP → `oof_mlp.npy`; hoàn thiện `blend.py` trước 16/10 |

Lịch tổng thể:

| Giai đoạn | Ngày | Nội dung |
| --- | --- | --- |
| Khởi động | 12-13/10 | Data, `folds.csv`, repo |
| Baseline | 13-16/10 | LightGBM, MLP, EDA, nộp baseline đầu |
| Cải thiện | 17-27/10 | Feature engineering, XGBoost/CatBoost, tuning, dữ liệu gốc, model đa dạng |
| Ensemble | 25-28/10 | Blend/stacking, chạy nhiều seed |
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
