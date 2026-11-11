# Tổng Quan Các File Trong Thư Mục `src/`

Thư mục `src/` chứa toàn bộ mã nguồn dùng chung của nhóm trong cuộc thi **Kaggle Playground Series S6E10**. Các file này được thiết kế để đảm bảo:
1. **Dùng chung một bộ fold duy nhất** (`StratifiedKFold`, seed 42) để kết quả cross-validation (CV) của mọi thành viên có thể so sánh và ghép vào Stacking được.
2. **Quy chuẩn hóa cách lưu trữ và kiểm tra dự đoán** (`oof_<tên>.npy` và `test_<tên>.npy`), chống data leakage.
3. **Cung cấp công cụ Stacking / Blending** đa mô hình tự động và xuất file nộp bài chuẩn quy định Kaggle.

---

## 1. Sơ đồ luồng hoạt động (Workflow Architecture)

```
                       ┌──────────────────────┐
                       │    src/config.py     │ (Đường dẫn, SEED=42, 21 Features)
                       └──────────┬───────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ src/make_folds.py│    │   src/data.py    │    │   src/utils.py   │
│  (Chỉ chạy 1 lần │    │ load_data()      │    │ save_preds()     │
│   tạo folds.csv) │    │ -> (train, test) │    │ load_preds()     │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         └───────────► Gắn fold ◄┘                       │
                           │                             │
                           ▼                             │
                ┌─────────────────────┐                  │
                │  Notebook Cá Nhân   │                  │
                │ (Train model 5 fold)│                  │
                └──────────┬──────────┘                  │
                           │ Xuất OOF & Test             │
                           ▼                             │
                ┌─────────────────────┐                  │
                │    oof_<tên>.npy    │◄─────────────────┘
                │   test_<tên>.npy    │
                └──────────┬──────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
┌──────────────────┐               ┌──────────────────┐
│   src/blend.py   │               │  src/submit.py   │
│ Rank Blend /     │               │ Tạo CSV nộp bài  │
│ LR Stacking P247 │               │ từ test_<tên>.npy│
└──────────────────┘               └──────────────────┘
```

---

## 2. Chi tiết từng file trong `src/`

### 📄 1. [`src/config.py`](file:///d:/Luan/Study/CS116/cs116-kaggle-s6e10/src/config.py) — Cấu hình tập trung
* **Vai trò:** Định nghĩa đường dẫn và danh mục đặc trưng cố định cho toàn bộ dự án.
* **Các hằng số quan trọng:**
  * `ROOT`: Đường dẫn thư mục gốc repo.
  * `DATA_DIR`, `OOF_DIR`, `SUB_DIR`: Tự động trỏ về `data/`, `oof/`, `submissions/` (cho phép ghi đè qua biến môi trường khi chạy trên Kaggle/Colab).
  * `N_FOLDS = 5`, `SEED = 42`: Cố định 5-fold cross-validation với seed 42.
  * `ID = "id"`, `TARGET = "satisfaction"`.
  * `CAT_COLS`: 4 cột phân loại (`Gender`, `Customer Type`, `Type of Travel`, `Class`).
  * `SCORE_COLS`: 13 cột khảo sát điểm dịch vụ từ 0 đến 5.
  * `NUM_COLS`: 4 cột số (`Age`, `Flight Distance`, `Departure Delay in Minutes`, `Arrival Delay in Minutes`).
  * `FEATURES`: Danh sách đầy đủ 21 cột đặc trưng (`CAT_COLS + SCORE_COLS + NUM_COLS`).

---

### 📄 2. [`src/data.py`](file:///d:/Luan/Study/CS116/cs116-kaggle-s6e10/src/data.py) — Nạp dữ liệu & Gắn Fold chuẩn
* **Vai trò:** Đọc dữ liệu tập train/test và tự động ánh xạ cột `fold`.
* **Hàm chính: `load_data()`**
  * Nạp `train.csv` và `test.csv`.
  * Chuyển nhãn `satisfaction` sang dạng số nguyên `0` hoặc `1`.
  * Đọc `data/folds.csv` và kiểm tra tính toàn vẹn thứ tự ID:
    ```python
    assert folds[ID].equals(train[ID]), "folds.csv không khớp thứ tự id của train.csv"
    ```
  * Gắn cột `train["fold"]` (giá trị từ `0` đến `4`).
  * **Trả về:** Cặp DataFrame `(train, test)`.

---

### 📄 3. [`src/utils.py`](file:///d:/Luan/Study/CS116/cs116-kaggle-s6e10/src/utils.py) — Quản lý & Kiểm tra dự đoán OOF / Test
* **Vai trò:** Đảm bảo tất cả các model được lưu đúng kích thước, không bị thiếu khuyết dữ liệu và in kiểm tra điểm CV AUC.
* **Các hằng số kiểm tra kích thước:**
  * `N_TRAIN = 699_635` (số dòng tập train).
  * `N_TEST = 299_844` (số dòng tập test).
* **Các hàm:**
  * `save_preds(name, oof, test_pred, y=None)`:
    * Kiểm tra mảng `oof` đúng shape `(699635,)` và `test_pred` đúng shape `(299844,)`.
    * Kiểm tra không chứa giá trị `NaN`.
    * Lưu thành 2 file: `oof/oof_<name>.npy` và `oof/test_<name>.npy`.
    * Nếu truyền nhãn `y`, hàm sẽ tự tính và in `CV AUC = roc_auc_score(y, oof)`.
  * `load_preds(name)`:
    * Đọc và trả về cặp mảng `(oof, test_pred)` của model `<name>` từ thư mục `oof/`.

---

### 📄 4. [`src/make_folds.py`](file:///d:/Luan/Study/CS116/cs116-kaggle-s6e10/src/make_folds.py) — Bộ sinh Fold (Chạy một lần duy nhất)
* **Vai trò:** Khởi tạo file phân chia `data/folds.csv`.
* **Cách hoạt động:**
  * Sử dụng `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` trên nhãn `train["satisfaction"]`.
  * **Cơ chế chống ghi đè an toàn:**
    ```python
    if out.exists():
        raise SystemExit(f"{out} đã tồn tại. Không tạo lại để tránh lệch fold giữa các thành viên.")
    ```
  * In bảng chéo tỷ lệ nhãn trên từng fold để đảm bảo tính phân tầng (stratified) đồng đều.

---

### 📄 5. [`src/blend.py`](file:///d:/Luan/Study/CS116/cs116-kaggle-s6e10/src/blend.py) — Động cơ Stacking & Blending
* **Vai trò:** Tự động kết hợp các model đơn lẻ thành model ensemble mạnh nhất thông qua Rank-Average hoặc Logistic Regression Stacking.
* **Bộ lọc kiểm định OOF tự động (Data Sanity Checks):**
  * Tự động loại model có $AUC < 0.90$ (nghi ghép sai thứ tự dòng hoặc nhãn).
  * Loại model có $AUC > 0.962$ (nghi là bản stack fit in-sample gây leak điểm CV).
  * Loại bỏ các model gần như trùng lặp (Rank correlation $\rho > 0.9999$ so với model có AUC cao hơn).
* **Hai chế độ kết hợp:**
  1. **Rank Blending (`--method rank`):**
     * Đưa xác suất về thứ hạng phân vị (`rankdata / N`) rồi tính trung bình có trọng số (`--weights`).
  2. **Logistic Regression Stacking (`--method lr`):**
     * Đưa dự đoán của từng member về thang đo **logit**.
     * Huấn luyện mô hình Logistic Regression ($C=0.003$) trên ma trận logit OOF theo từng fold.
     * Hỗ trợ **Nested CV** qua nhiều meta-seed (`--meta-seed 7 101`) để đảm bảo điểm tăng là thật, không phải may rủi.
* **Cách sử dụng dòng lệnh:**
  ```bash
  # Xem tương quan và rank blend 2 model:
  python src/blend.py lgb mlp

  # Chạy LR Stack trên toàn bộ file OOF sạch trong oof/:
  python src/blend.py --method lr --C 0.003

  # Đối chiếu độ ổn định của Stacker trên nhiều seed và lưu kết quả:
  python src/blend.py --method lr --meta-seed 7 101 --save stack_v1
  ```

---

### 📄 6. [`src/submit.py`](file:///d:/Luan/Study/CS116/cs116-kaggle-s6e10/src/submit.py) — Xuất file nộp bài Kaggle
* **Vai trò:** Chuyển đổi file dự đoán `test_<name>.npy` thành file CSV nộp lên Kaggle và kiểm tra định dạng.
* **Kiểm tra an toàn:**
  * Khớp số lượng dòng với `sample_submission.csv`.
  * Không chứa giá trị `NaN`.
  * Bắt buộc dự đoán là **xác suất liên tục** (chặn nhãn cứng nhị phân `0/1` vì metric là ROC AUC).
* **Kết quả:**
  * Lưu tại `submissions/sub_<name>.csv`.
  * In ra câu lệnh Kaggle CLI để nộp nhanh:
    ```bash
    kaggle competitions submit -c playground-series-s6e10 -f "submissions/sub_name.csv" -m "name"
    ```

---

## 3. Hướng dẫn nhúng `src/` vào Notebook cá nhân

Để tận dụng toàn bộ pipeline dùng chung trong bất kỳ notebook nào (ví dụ `notebooks/cs116_llr_oof_local.ipynb`), chỉ cần thêm đoạn mã ở đầu notebook:

```python
import sys
from pathlib import Path

# Thêm src/ vào sys.path
SRC_DIR = Path.cwd().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import trực tiếp các module chuẩn của nhóm
from config import DATA_DIR, OOF_DIR, FEATURES, ID, TARGET, SEED
from data import load_data
from utils import save_preds, load_preds

# 1. Nạp dữ liệu và fold chuẩn:
train, test = load_data()
y = train[TARGET].to_numpy()
folds = train["fold"].to_numpy()

# 2. Sau khi train xong 5 fold:
save_preds("my_model_v1", oof_preds, test_preds, y=y)
```
Sau khi chạy xong, file `oof/oof_my_model_v1.npy` và `oof/test_my_model_v1.npy` sẽ được tạo đúng chuẩn, sẵn sàng để nạp vào `src/blend.py` hoặc hệ thống Stack v7.
