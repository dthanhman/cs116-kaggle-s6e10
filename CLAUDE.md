# Đồ án CS116: Kaggle Playground Series S6E10 - Predicting Airline Satisfaction

> File này giúp Claude nắm toàn bộ đồ án. Cập nhật lần cuối: 12/10/2026. Mục nào ghi `CHƯA RÕ` là chưa có thông tin, đừng tự bịa.

## 1. Mục tiêu

- Mục tiêu của nhóm: **vào top 5%** bảng xếp hạng cuộc thi.
- Cuộc thi: [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10), tên "Predicting Airline Satisfaction".
- Nhóm 3 người, vai trò A / B / C (xem mục 5). Tên thật: `CHƯA RÕ` (điền vào bảng ở mục 5).

## 2. Thông tin cuộc thi

| Mục | Nội dung |
| --- | --- |
| Bài toán | Phân loại nhị phân: dự đoán xác suất hành khách hài lòng (`satisfaction`) |
| Metric | **AUC** (diện tích dưới đường ROC) giữa xác suất dự đoán và nhãn thật |
| Bắt đầu | 1/10/2026 |
| Hạn nộp cuối | **31/10/2026 23:59 UTC** = 6:59 sáng 1/11/2026 giờ Việt Nam |
| Hạn gộp team, hạn vào thi | Trùng với hạn nộp cuối |
| Giải thưởng | Quà Kaggle cho top 3 (không liên quan mục tiêu top 5%) |
| Số lần submit mỗi ngày | `CHƯA RÕ` (xem tab Submissions; tạm giả định 5 lần/ngày) |
| Số bài được chọn làm bài cuối | `CHƯA RÕ` (kế hoạch chọn 2 bài) |

## 3. Dữ liệu

- File: `train.csv` (có cột target `satisfaction`), `test.csv`, `sample_submission.csv`.
- Dữ liệu **tổng hợp**, lấy cảm hứng từ bộ gốc [binary-aviation-satisfaction-129k](https://www.kaggle.com/datasets/arseniyshutko/binary-aviation-satisfaction-129k) (khoảng 129 nghìn dòng). Phân phối gần giống nhưng không giống hệt bản gốc.
- Định dạng nộp: có header, hai cột `id,satisfaction`, giá trị là **xác suất** (không phải nhãn 0/1).
  ```
  id,satisfaction
  699635,0.2
  699636,0.3
  ```
- Đã kiểm tra 12/10:
  - Train 699.635 dòng (id 0 đến 699634), test 299.844 dòng (id 699635 đến 999478).
  - Target kiểu bool: True 44,36%.
  - Chỉ `Arrival Delay in Minutes` có thiếu (292 dòng train, 130 dòng test).
  - 21 feature: 4 category (`Gender`, `Customer Type`, `Type of Travel`, `Class`), 13 điểm dịch vụ 0-5, 4 cột số (`Age`, `Flight Distance`, 2 cột delay).
  - Delay phần lớn bằng 0, trung vị 0.
  - Không có dòng trùng, id không leak, train và test cùng phân phối.
- `data/folds.csv`: 5 fold, StratifiedKFold, seed 42. Baseline logreg CV AUC 0,92155.

## 4. Nguyên tắc kỹ thuật chung

1. **Một bộ fold duy nhất**: `folds.csv` do A tạo (StratifiedKFold, seed cố định). Không ai tự chia fold riêng.
2. **Lưu kết quả mỗi model**: `oof_<tên>.npy` và `test_<tên>.npy`, cùng thứ tự dòng với `train.csv` và `test.csv`.
3. **Metric là AUC** nên chỉ thứ hạng xác suất quan trọng: không chỉnh ngưỡng, blend bằng **rank average** hoặc trọng số tối ưu trên OOF AUC.
4. **Dữ liệu gốc (129k dòng)**: chỉ thêm vào các fold *train*, tuyệt đối không đưa vào fold *validation* (nếu không CV sẽ lạc quan giả). Có dùng hay không phải quyết bằng CV.
5. **Tin CV hơn LB công khai.** Nếu CV và LB lệch nhiều, A điều tra (adversarial validation, leak) trước khi cả nhóm đổi hướng.
6. **Log thí nghiệm**: mỗi model một dòng gồm tên model, CV AUC, LB AUC, ngày, người làm.
7. **Nộp bài**: chỉ A nộp lên Kaggle cho cả nhóm để kiểm soát số lần submit.
8. **Hai bài cuối**: một bản ensemble tốt nhất theo CV, một bản ổn định và đa dạng hơn.

## 5. Phân vai

| Người | Tên thật | Vai trò | Trách nhiệm chính |
| --- | --- | --- | --- |
| A | `CHƯA RÕ` | Data & Validation Lead, kiêm Tech Lead | EDA, `folds.csv`, adversarial validation, thử dữ liệu gốc, repo, script submit, theo dõi CV vs LB, nộp bài |
| B | `CHƯA RÕ` | GBDT & Feature Engineering Lead | LightGBM, XGBoost, CatBoost, feature engineering, tuning Optuna |
| C | `CHƯA RÕ` | Neural Net & Ensemble Lead | MLP tabular, model đa dạng (logistic, ExtraTrees), rank-blend, stacking, chốt 2 bài cuối |

Hỗ trợ chéo: A xong phần nền thì giúp nhánh đang tụt; B và C báo feature, ý tưởng có ích cho nhau; họp 15 phút mỗi 2-3 ngày để đổi việc nếu một nhánh kẹt.

MLP là **giả thuyết cần kiểm chứng** (tăng đa dạng ensemble), không phải mặc định. Nếu thêm OOF của MLP mà CV AUC của blend không tăng đáng kể sau vài ngày, C chuyển sang model đa dạng rẻ hơn hoặc giúp B tuning GBDT.

## 6. Lịch và danh sách nhiệm vụ

Nằm ở [docs/KE_HOACH.md](docs/KE_HOACH.md) (nguồn duy nhất, chỉ sửa ở đó). Mốc quan trọng: đóng băng model 29/10, hạn nộp 6:59 sáng 1/11 giờ Việt Nam.

## 7. Cách Claude nên giúp

- Trả lời bằng **tiếng Việt**, ngắn gọn, rõ ràng, tránh lan man.
- Luôn bám metric AUC và mục tiêu top 5%; ưu tiên việc có khả năng tăng CV AUC nhất.
- Không nộp nhãn 0/1, không đưa dữ liệu gốc vào fold validation, không dùng fold khác `folds.csv`.
- Khi viết code, lưu OOF và test preds đúng quy ước ở mục 4, ghi rõ CV AUC.
- Mục `CHƯA RÕ` thì hỏi người dùng hoặc nói rõ là giả định, không tự điền.
- Với thông tin về cuộc thi có thể đã đổi (hạn, luật, số lần submit), nhắc kiểm tra lại trên trang Kaggle.

## 8. Việc còn thiếu

- [ ] Điền tên thật cho A / B / C (mục 5).
- [ ] Xác nhận số lần submit mỗi ngày và số bài chọn làm bài cuối.
- [ ] A cập nhật mục 3 sau khi tải dữ liệu: số dòng, tên cột, giá trị thiếu, tỉ lệ target.
- [ ] Ghi nơi chạy code (Kaggle Notebook, Colab, máy cá nhân), có GPU không.
- [ ] Ghi điểm CV và LB của baseline đầu tiên.
