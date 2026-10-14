# Kế hoạch chung: mỗi người làm OOF, cuối cùng stack

## 1. Ý tưởng trong một câu

Mỗi người tự xây và cải thiện model của mình, **dùng chung một bộ fold** (`data/folds.csv`), lưu dự đoán theo **cùng một quy ước**. Cuối cùng, dự đoán của mọi người được **stack** lại thành bài nộp.

- Việc stack và nộp bài do một người phụ trách. Mọi người **không tự nộp lên Kaggle** (để giữ số lượt nộp mỗi ngày); muốn thử một bài thì gửi tên file cho người đó.
- Những người còn lại chỉ cần tập trung làm ra **OOF tốt và sạch** theo mục 5.

## 2. Thuật ngữ cần biết

| Từ | Nghĩa |
| --- | --- |
| **Nhãn** | Đáp án thật của mỗi dòng train: cột `satisfaction` (hài lòng hay không) |
| **AUC** | Thước đo của cuộc thi, từ 0,5 (đoán bừa) đến 1 (hoàn hảo). Chỉ quan tâm model có xếp người hài lòng lên trên người không hài lòng hay không |
| **Fold** | Train được chia sẵn thành 5 phần bằng nhau gọi là 5 fold (ghi trong `data/folds.csv`, cột `fold` từ 0 đến 4) |
| **Fold validation** | Fold đang được giấu đi để chấm điểm. Model train trên 4 fold còn lại rồi dự đoán fold này |
| **CV** (cross-validation) | Làm như trên 5 lần, mỗi fold được giấu một lần. **CV AUC** là AUC tính trên cả 5 fold đó. Đây là điểm chính để so sánh model |
| **OOF** (out-of-fold) | Dự đoán cho mỗi dòng train, lấy **lúc dòng đó nằm trong fold validation**, tức lúc model chưa thấy nhãn của nó. Gồm 699.635 giá trị, cùng thứ tự với `train.csv` |
| **Test preds** | Dự đoán cho 299.844 dòng của `test.csv`, thường là trung bình của 5 model ở 5 fold |
| **LB** (leaderboard) | Bảng xếp hạng Kaggle. **Public LB** chỉ chấm khoảng 20% test nên dao động nhiều; **private LB** chấm phần còn lại và là điểm xếp hạng cuối cùng |
| **Early stopping** | Dừng thêm cây khi điểm trên một tập kiểm tra không tăng nữa, để chọn số cây |
| **Dữ liệu gốc** | Bộ 129.880 dòng thật mà dữ liệu cuộc thi được sinh ra từ đó. Tải về `data/original/` bằng lệnh `kaggle datasets download -d arseniyshutko/binary-aviation-satisfaction-129k --unzip -p data/original` |
| **Teacher** | Một model **chỉ train trên dữ liệu gốc**, rồi dùng xác suất nó dự đoán cho train/test làm một feature. Nó không thấy nhãn của cuộc thi nên không gây leak |
| **Leak** | Model vô tình được thấy nhãn của dòng đang chấm điểm, khiến CV cao giả rồi tụt khi nộp |

## 3. Stack là gì?

Giả sử có 3 model cùng dự đoán xác suất hài lòng của 3 hành khách:

| Hành khách | Model 1 | Model 2 | Model 3 | Nhãn |
| --- | --- | --- | --- | --- |
| An | 0,90 | 0,85 | 0,70 | Hài lòng |
| Bình | 0,40 | 0,20 | 0,60 | Không |
| Chi | 0,55 | 0,75 | 0,50 | Hài lòng |

Có 3 cách gộp chúng lại:

1. **Lấy trung bình**: đơn giản, nhưng coi mọi model ngang nhau.
2. **Trung bình có trọng số**: tự chọn, ví dụ 50% / 30% / 20%. Chọn bằng tay thì khó.
3. **Stack**: coi 3 cột dự đoán là 3 feature mới, cột nhãn vẫn là nhãn, rồi **train thêm một model nhỏ** (thường là logistic regression) để nó **tự học nên tin model nào bao nhiêu**.

Model nhỏ ở cách 3 gọi là **stacker**. Các model ban đầu gọi là **member**. Stacker học trên **OOF** của các member, rồi áp dụng lên **test preds** của chúng để ra bài nộp.

**Vì sao stack giúp tăng điểm?** Mỗi model sai ở những chỗ khác nhau. Kết hợp khéo thì các lỗi bù trừ cho nhau. Trong cuộc thi này:

| Cách làm | CV AUC |
| --- | --- |
| Model đơn tốt nhất | khoảng 0,9611–0,9617 |
| Stack khoảng 250 model mà người chơi khác đã công khai OOF trên Kaggle | khoảng 0,9622 |

Chênh 0,0006 nghe nhỏ, nhưng ở cuộc thi này đó là khoảng cách giữa ngoài và trong top 5% (ngưỡng ngày 13/10: LB 0,96186).

## 4. Vì sao OOF phải "sạch" và dùng chung fold

**Phải sạch:** stacker học xem member nào đáng tin dựa trên OOF. Nếu một member đã thấy nhãn của dòng nó dự đoán, nó sẽ đoán đúng một cách giả tạo, stacker tin nhầm nó, CV cao giả rồi tụt khi nộp. Giống một học sinh đã xem trước đáp án đề thi thử: điểm thi thử cao nhưng thi thật thì không.

**Phải chung fold:** khi stacker giấu một fold để tự chấm điểm, mọi member đều phải "chưa thấy" đúng fold đó. Nếu mỗi người chia fold một kiểu, member này chưa thấy dòng X nhưng member kia đã thấy, và stack bị leak.

Bộ fold của nhóm (StratifiedKFold 5 fold, `shuffle=True`, `random_state=42`) trùng với hầu hết OOF công khai trên Kaggle, nên OOF của nhóm còn ghép được với cả bộ công khai đó.

## 5. Chuẩn của một OOF "sạch"

Trước khi gửi một OOF, kiểm tra đủ 6 điều.

### Điều 1: dùng đúng fold chung

Đọc dữ liệu bằng `load_data()` trong `src/data.py`. Hàm này gắn sẵn cột `fold` lấy từ `folds.csv`. **Không tự chia fold** bằng `StratifiedKFold` hay `train_test_split`.

```python
from data import load_data
train, test = load_data()          # train có cột 'fold' (0..4), nhãn đã đổi sang 0/1
for k in range(5):
    tr = train["fold"] != k        # 4 fold để train
    va = train["fold"] == k        # fold k để chấm điểm
```

### Điều 2: feature tính từ nhãn phải tính trong fold

Feature nào **dùng nhãn để tính** (ví dụ target encoding: thay mỗi giá trị bằng tỉ lệ hài lòng của nhóm có giá trị đó) thì trong mỗi vòng fold `k` chỉ được tính từ **phần train** (`tr`), rồi áp sang fold `k` và test. Nếu tính trên toàn bộ train, giá trị của dòng ở fold `k` đã chứa nhãn của chính nó, tức là leak.

Feature **không dùng nhãn** (đếm tần suất, tách chữ số, cộng trừ các cột…) thì tính một lần trên cả train và test là được.

Tuning tham số (ví dụ bằng Optuna) dựa trên CV với `folds.csv` thì **được**, không cần làm riêng trong từng fold.

### Điều 3: early stopping không dùng fold đang chấm điểm

Nếu dừng sớm theo chính fold `k`, ta đã chọn số cây sao cho điểm trên fold `k` đẹp nhất, nên CV hơi cao giả. Bài nộp trước đây của nhóm có CV lệch LB khoảng 0,0005, một phần do việc này. Chọn một trong hai cách:

**Cách a: tách 10% từ phần train của fold để dừng sớm.**

```python
import numpy as np
tr_idx = np.where(train["fold"] != k)[0]
rng = np.random.default_rng(42)
es_idx = rng.choice(tr_idx, size=len(tr_idx) // 10, replace=False)   # 10% để dừng sớm
fit_idx = np.setdiff1d(tr_idx, es_idx)                                # 90% còn lại để train
# train trên fit_idx, early stopping trên es_idx, rồi dự đoán fold k và test
```

**Cách b: cố định số cây.** Chạy thử một lần theo cách a, lấy số cây trung bình của 5 fold, rồi các lần sau dùng đúng số cây đó, không early stopping nữa.

### Điều 4: dữ liệu gốc không được vào fold chấm điểm

Dữ liệu gốc chỉ được dùng theo hai cách:

- **Làm feature**: teacher (xem mục 2), hoặc tra tỉ lệ hài lòng theo từng giá trị trên dữ liệu gốc.
- **Ghép thêm dòng vào phần train của fold** (4 fold dùng để train). Fold đang chấm điểm vẫn chỉ có dữ liệu cuộc thi.

### Điều 5: chỉ gửi model gốc

Gửi dự đoán của **từng model riêng lẻ**. Không gửi bản đã blend hay stack, kể cả blend do mình tự làm. Lý do xem mục 8.

### Điều 6: lưu đúng quy ước và tự kiểm tra

Lưu bằng `save_preds()`. Hàm này tự kiểm tra số dòng, NaN, và in CV AUC:

```python
from utils import save_preds
save_preds("A_lgb_v1", oof, test_pred, y)   # -> oof/oof_A_lgb_v1.npy, oof/test_A_lgb_v1.npy
```

Sau đó kiểm tra lại bằng lệnh (chạy trong thư mục `src/`, hai file phải nằm trong thư mục `oof/`):

```
cd src
python blend.py A_lgb_v1
```

Lệnh in CV AUC của model. AUC hợp lý của cuộc thi này nằm trong khoảng **0,90 đến 0,962**:

- Dưới 0,90: gần như chắc chắn bị lệch thứ tự dòng (ví dụ lỡ xáo trộn dòng) hoặc sai nhãn.
- Trên 0,962: cao hơn mọi model đơn đã biết, nhiều khả năng có leak.

Lệnh sẽ tự báo nếu model rơi vào một trong hai trường hợp này.

## 6. Gửi OOF cho nhóm

1. **Đặt tên** dạng `<người>_<model>_<phiên bản>`, ví dụ `A_lgb_v1`, `B_cat_v2`, `C_mlp_v1`. Tên này dùng cho cả `save_preds()` và mọi chỗ khác. **Không dùng tên bắt đầu bằng `blend_` hay `stack_`** (dành cho người stack).
2. **Ghi một dòng vào `experiments.csv`** theo thứ tự cột: ngày, người, tên, CV AUC, LB AUC (để trống), ghi chú ngắn. Ví dụ:
   ```
   2026-10-16,B,B_cat_v1,0.96085,,CatBoost mọi cột là category, early stopping 10%
   ```
3. **Upload 2 file** `oof_<tên>.npy` và `test_<tên>.npy` lên thư mục `oof/` trên Drive chung.
4. **Commit code** đã tạo ra OOF đó, để người khác chạy lại được.

## 7. Member như thế nào thì có ích cho stack

- **Khác biệt quan trọng hơn mạnh.** Một model sai ở chỗ các model khác không sai thì có ích, dù AUC riêng thấp hơn một chút. Mười bản LightGBM gần giống nhau (dự đoán tương quan 0,99) thì gần như chỉ bằng một.
- Cách tạo khác biệt: **loại model khác** (cây, tuyến tính, mạng nơ-ron, kNN…), **bộ feature khác**, **cách dùng dữ liệu gốc khác**.
- Bộ OOF công khai đã có khoảng 160 model cây (LightGBM, XGBoost, CatBoost), 58 mạng nơ-ron, 16 TabPFN. Thêm một model cây dùng feature phổ biến gần như không giúp. Member của nhóm có giá trị nhất khi dùng **ý tưởng người khác chưa có**.
- **Tiêu chí giữ một member**: thêm nó vào thì điểm của **stack** tăng ổn định, không phải AUC riêng của nó cao. Người stack sẽ đo việc này và báo lại.

**Đo điểm của stack thế nào cho trung thực?** Bản thân stacker cũng được chấm bằng CV: giấu một fold, train stacker trên OOF của 4 fold còn lại, chấm trên fold đã giấu. Vì stacker được train và chấm trên OOF (vốn đã là kết quả của CV), đây là CV lồng trong CV, gọi là **nested CV**. Sau đó làm lại với vài cách chia fold khác cho stacker (đổi seed) để chắc chắn mức tăng không phải may mắn.

## 8. Những bẫy cần tránh

| Bẫy | Hậu quả |
| --- | --- |
| Đưa blend/stack có sẵn (của nhóm hoặc người khác) vào làm member | Stacker đó đã được train trên mọi dòng, kể cả fold đang chấm, tức là đã thấy nhãn. Stack mới dựa vào nó nên CV cao giả mà LB không tăng. Một notebook công khai của cuộc thi đo được mức cao giả tới +0,00004 |
| Tự chia fold khác | OOF không ghép được với của người khác, stack bị leak |
| Chọn bài theo public LB | Public LB chỉ chấm khoảng 20% test, dao động khoảng ±0,00002. Chọn bản có public LB cao nhất thực chất là chọn bản may mắn với phần test đó, dễ tụt khi private LB công bố |
| Thử nhiều cấu hình rồi chỉ giữ cái có CV cao nhất, kể cả việc bỏ bớt member vì "thấy thừa" | Giống chọn người điểm cao nhất kỳ thi thử: điểm đó luôn có chút may mắn. Một người chơi thử bỏ 48 member: CV tăng nhẹ nhưng LB giảm. Một thay đổi chỉ đáng tin khi tăng ở phần lớn fold và tăng lại khi đổi seed |

## 9. Mốc thời gian

| Ngày | Mốc |
| --- | --- |
| 15–17/10 | Mỗi người có **ít nhất 1 OOF sạch** gửi lên Drive |
| 18–24/10 | Phát triển thêm member khác biệt. **21/10**: chạy stack giữa kỳ, báo lại member nào có ích |
| 22–27/10 | Ghép OOF của nhóm với bộ OOF công khai, đo đóng góp của từng member |
| 28/10 | Chạy lại các member được giữ với nhiều seed rồi lấy trung bình |
| **29/10** | **Đóng băng**: không thêm member mới. Chốt 2 bài cuối |
| 30–31/10 | Chỉ nộp và xác nhận |

**Hai bài cuối:** một bản stack tốt nhất theo nested CV, một bản ổn định hơn (ít thay đổi, ít rủi ro).
