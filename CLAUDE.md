# Đồ án CS116: Kaggle Playground Series S6E10 - Predicting Airline Satisfaction

> File này giúp Claude nắm toàn bộ đồ án. Cập nhật lần cuối: 14/10/2026. Mục nào ghi `CHƯA RÕ` là chưa có thông tin, đừng tự bịa.

## 1. Mục tiêu

- Mục tiêu của nhóm: **vào top 5%** bảng xếp hạng cuộc thi.
- Cuộc thi: [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10), tên "Predicting Airline Satisfaction".
- Nhóm 3 người, vai trò A / B / C. Tên thật: `CHƯA RÕ`.

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
2. **Lưu kết quả mỗi model**: `oof_<tên>.npy` và `test_<tên>.npy` qua `save_preds()`, cùng thứ tự dòng với `train.csv` và `test.csv`. Tên dạng `<người>_<model>_<phiên bản>` (ví dụ `A_lgb_v1`); tên `blend_*` / `stack_*` dành cho kết quả blend/stack.
3. **OOF phải sạch** (chi tiết 6 điều kiện ở `docs/STACKING.md` mục 5): feature tính từ nhãn (target encoding…) tính trong fold; **early stopping không dùng fold đang chấm điểm** (tách 10% từ phần train của fold hoặc cố định số cây); chỉ gửi OOF của model gốc, không đưa blend/stack làm member.
4. **Dữ liệu gốc (129k dòng, `data/original/data.csv`)**: chỉ dùng làm feature (teacher chỉ train trên dữ liệu gốc, tra cứu tỉ lệ) hoặc thêm vào phần train của fold; tuyệt đối không đưa vào fold *validation*. Có dùng hay không phải quyết bằng CV.
5. **Metric là AUC** nên chỉ thứ hạng xác suất quan trọng: không chỉnh ngưỡng. Kết hợp model bằng stack LR trên logit OOF (`src/blend.py --method lr`), đánh giá bằng nested CV qua nhiều meta seed.
6. **Tin CV hơn LB công khai.** Một thay đổi chỉ được giữ khi tăng ở phần lớn fold và qua nhiều seed. Nếu CV và LB lệch nhiều, điều tra leak trước khi cả nhóm đổi hướng.
7. **Log thí nghiệm**: mỗi model một dòng trong `experiments.csv` (ngày, người, tên, CV AUC, LB AUC, ghi chú).
8. **Nộp bài**: chỉ một người nộp lên Kaggle cho cả nhóm để kiểm soát số lần submit.
9. **Hai bài cuối**: một bản stack tốt nhất theo nested CV, một bản ổn định hơn.

## 5. Cách phối hợp

Mỗi người tự xây và cải thiện model của mình trên `folds.csv`, gửi OOF sạch (tên, `experiments.csv`, Drive); cuối cùng một người stack mọi OOF của nhóm, kèm bộ OOF công khai trên Kaggle (khoảng 247 member cùng fold seed 42), thành bài nộp. Member được giữ khi làm stack tăng ổn định, không phải khi AUC riêng cao; ưu tiên member **khác biệt** (loại model, bộ feature, cách dùng dữ liệu gốc khác) vì bộ công khai đã bão hòa model cây.

Tài liệu cho nhóm: [docs/STACKING.md](docs/STACKING.md) (giải thích stack, chuẩn OOF sạch, mốc thời gian). Thông tin cuộc thi và việc còn thiếu: [docs/KE_HOACH.md](docs/KE_HOACH.md).

## 6. Mốc thời gian

15–17/10 mỗi người có OOF sạch đầu tiên; 21/10 stack giữa kỳ; 22–27/10 ghép với OOF công khai; 28/10 chạy nhiều seed; **29/10 đóng băng**, chốt 2 bài; hạn nộp 6:59 sáng 1/11 giờ Việt Nam.

## 7. Cách Claude nên giúp

- Trả lời bằng **tiếng Việt**, ngắn gọn, rõ ràng, tránh lan man.
- Luôn bám metric AUC và mục tiêu top 5%; ưu tiên việc có khả năng tăng CV AUC nhất.
- Không nộp nhãn 0/1, không đưa dữ liệu gốc vào fold validation, không dùng fold khác `folds.csv`.
- Khi viết code, lưu OOF và test preds đúng quy ước ở mục 4, ghi rõ CV AUC.
- Mục `CHƯA RÕ` thì hỏi người dùng hoặc nói rõ là giả định, không tự điền.
- Với thông tin về cuộc thi có thể đã đổi (hạn, luật, số lần submit), nhắc kiểm tra lại trên trang Kaggle.

## 8. Việc còn thiếu

- [ ] Điền tên thật cho A / B / C.
- [ ] Xác nhận số lần submit mỗi ngày và số bài chọn làm bài cuối.
- [x] A cập nhật mục 3 sau khi tải dữ liệu: số dòng, tên cột, giá trị thiếu, tỉ lệ target.
- [ ] Ghi nơi chạy code (Kaggle Notebook, Colab, máy cá nhân), có GPU không.
- [x] Ghi điểm CV và LB của bài nộp đầu tiên (xem `experiments.csv`; ngưỡng top 5% ngày 13/10: LB 0,96186).
