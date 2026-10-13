# Kế hoạch của nhóm

Kế hoạch chung cho cuộc thi [Playground Series S6E10: Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10). Hướng dẫn cài đặt và quy trình làm việc nằm ở [README](../README.md). Lịch và nhiệm vụ chỉ sửa ở file này.

> Cập nhật lần cuối: 12/10/2026. Mục ghi `CHƯA RÕ` là chưa có thông tin, cần điền.

## 1. Mục tiêu

- Vào **top 5%** bảng xếp hạng.
- Nhóm 3 người, vai trò A / B / C. Tên thật: `CHƯA RÕ`.

## 2. Thông tin cuộc thi

| Mục                           | Nội dung                                                                  |
| ----------------------------- | ------------------------------------------------------------------------- |
| Bài toán                      | Phân loại nhị phân: dự đoán xác suất hành khách hài lòng (`satisfaction`) |
| Metric                        | **AUC** giữa xác suất dự đoán và nhãn thật                                |
| Bắt đầu                       | 1/10/2026                                                                 |
| Hạn nộp cuối                  | **31/10/2026 23:59 UTC** = 6:59 sáng 1/11/2026 giờ Việt Nam               |
| Hạn gộp team, hạn vào thi     | Trùng với hạn nộp cuối                                                    |
| Số lần submit mỗi ngày        | `CHƯA RÕ` (xem tab Submissions; tạm giả định 5 lần/ngày)                  |
| Số bài được chọn làm bài cuối | `CHƯA RÕ` (kế hoạch chọn 2 bài)                                           |

Hạn, luật và số lần submit có thể đổi, nên kiểm tra lại trên trang Kaggle.

## 3. Phân vai

| Người | Tên thật  | Vai trò                                | Trách nhiệm chính                                                                                          |
| ----- | --------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| A     | `CHƯA RÕ` | Data & Validation Lead, kiêm Tech Lead | EDA, `folds.csv`, adversarial validation, thử dữ liệu gốc, repo, script submit, theo dõi CV vs LB, nộp bài |
| B     | `CHƯA RÕ` | GBDT & Feature Engineering Lead        | LightGBM, XGBoost, CatBoost, feature engineering, tuning Optuna                                            |
| C     | `CHƯA RÕ` | Neural Net & Ensemble Lead             | MLP tabular, model đa dạng (logistic, ExtraTrees), rank-blend, stacking, chốt 2 bài cuối                   |

- **Hỗ trợ chéo:** A xong phần nền thì giúp nhánh đang tụt. B và C báo feature, ý tưởng có ích cho nhau.
- **MLP là giả thuyết cần kiểm chứng**, không phải mặc định. Nếu thêm OOF của MLP mà CV AUC của blend không tăng đáng kể sau vài ngày, C chuyển sang model đa dạng rẻ hơn hoặc giúp B tuning GBDT.

## 4. Lịch (giờ Việt Nam)

| Giai đoạn     | Ngày      | Nội dung                                                                   |
| ------------- | --------- | -------------------------------------------------------------------------- |
| Khởi động     | 12-13/10  | Tải data, tạo `folds.csv`, repo, script submit                             |
| Baseline      | 13-16/10  | LightGBM (B), MLP (C), EDA và adversarial validation (A), nộp baseline đầu |
| Cải thiện     | 17-27/10  | Feature engineering, tuning, thử dữ liệu gốc, model đa dạng                |
| Ensemble      | 25-28/10  | Blend/stacking trên OOF, chạy lại nhiều seed                               |
| **Đóng băng** | **29/10** | Không đổi model hay feature nữa; chốt 2 bài cuối                           |
| Dự phòng      | 30-31/10  | Chỉ nộp và xác nhận, không thử ý tưởng mới                                 |

## 5. Danh sách nhiệm vụ

**Trạng thái** (ai làm, xong chưa, còn bao nhiêu ngày) chỉ theo dõi trên [Google Sheet "Tasks"](https://docs.google.com/spreadsheets/d/1Ls0cM8Mt0s6bU7mPUonIcE_Wif6sw8_sCx9ApssyiuY/edit?usp=sharing). Bảng dưới đây chỉ là mô tả kế hoạch, không cập nhật trạng thái ở đây.

Lưu ý: số thứ tự trong sheet hiện chưa khớp với bảng dưới (sheet bỏ qua số 4). Khi nhắc tới nhiệm vụ, nên ghi kèm tên, ví dụ "#4 baseline LightGBM".

| #   | Người   | Nhiệm vụ                                                                         | Bắt đầu | Hạn   | Sản phẩm                             |
| --- | ------- | -------------------------------------------------------------------------------- | ------- | ----- | ------------------------------------ |
| 1   | A       | Tải dữ liệu, kiểm tra nhanh: số dòng, kiểu cột, giá trị thiếu, tỉ lệ target      | 12/10   | 13/10 | Ghi chú kích thước và tỉ lệ target   |
| 2   | A       | Tạo `folds.csv` (StratifiedKFold, seed cố định), gửi cả nhóm                     | 12/10   | 13/10 | `folds.csv`                          |
| 3   | A       | Tạo repo chung và script submit đúng định dạng                                   | 12/10   | 13/10 | Repo, `submit.py`                    |
| 4   | B       | Baseline LightGBM với `folds.csv`, ghi CV AUC                                    | 13/10   | 16/10 | `oof_lgb.npy`, `test_lgb.npy`        |
| 5   | C       | Baseline MLP với `folds.csv`, ghi CV AUC                                         | 13/10   | 16/10 | `oof_mlp.npy`, `test_mlp.npy`        |
| 6   | C       | Hàm rank-blend nhận nhiều OOF và tính AUC                                        | 13/10   | 16/10 | `blend.py`                           |
| 7   | A       | EDA và adversarial validation giữa train và test                                 | 14/10   | 16/10 | Báo cáo EDA ngắn                     |
| 8   | A       | Nộp baseline đầu lên Kaggle, ghi CV so với LB                                    | 16/10   | 16/10 | Dòng đầu trong bảng theo dõi         |
| 9   | A       | Thử thêm dữ liệu gốc, chỉ vào fold train                                         | 17/10   | 24/10 | Kết luận dùng hay không              |
| 10  | B       | Thêm XGBoost và CatBoost                                                         | 17/10   | 20/10 | OOF và test preds                    |
| 11  | B       | Feature engineering (nhóm điểm dịch vụ, delay, Class × Type of Travel, encoding) | 17/10   | 24/10 | Danh sách feature có ích             |
| 12  | C       | Cải thiện MLP, thêm model đa dạng                                                | 17/10   | 27/10 | OOF và test preds                    |
| 13  | A       | Tìm leak, theo dõi CV lệch LB, hỗ trợ nhánh tụt                                  | 17/10   | 27/10 | Ghi chú trong bảng theo dõi          |
| 14  | B       | Tuning Optuna cho GBDT tốt nhất                                                  | 21/10   | 27/10 | Bộ tham số tốt nhất                  |
| 15  | C       | Blend/stacking trên OOF cả nhóm                                                  | 25/10   | 28/10 | Bản ensemble tốt nhất theo CV        |
| 16  | B       | Chạy lại model tốt nhất với nhiều seed                                           | 27/10   | 28/10 | OOF và test preds đã trung bình seed |
| 17  | Cả nhóm | Đóng băng model                                                                  | 29/10   | 29/10 | Danh sách file cuối                  |
| 18  | C       | Chốt 2 submission cuối cùng với nhóm                                             | 29/10   | 29/10 | 2 file submit                        |
| 19  | A       | Kiểm tra định dạng, nộp thử 2 bài                                                | 29/10   | 29/10 | Xác nhận hợp lệ                      |
| 20  | A       | Nộp và xác nhận 2 bài cuối                                                       | 30/10   | 31/10 | 2 bài cuối trên Kaggle               |
| 21  | Cả nhóm | Họp 15 phút mỗi 2-3 ngày                                                         | 12/10   | 28/10 | Ghi chú họp                          |

**Hai bài cuối:** một bản ensemble tốt nhất theo CV, một bản ổn định và đa dạng hơn.

## 6. Việc còn thiếu

- [ ] Điền tên thật cho A / B / C.
- [ ] Xác nhận số lần submit mỗi ngày và số bài chọn làm bài cuối.
- [ ] Ghi nơi chạy code (Kaggle Notebook, Colab, máy cá nhân), có GPU không.
- [ ] Ghi điểm CV và LB của baseline đầu tiên (A nộp 16/10).
- [x] Tạo Drive chung, điền link vào README.
- [ ] Up `folds.csv` lên Drive, mời B và C vào repo, Drive và team Kaggle.
