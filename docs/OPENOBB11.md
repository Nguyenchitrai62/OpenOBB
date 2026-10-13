# OpenOBB11: đề xuất dense + bộ giải mã tập một-một (khử trùng tự học, không NMS) (2026-10-13)

`model=openobb11s | m | l | x`. CLI và API giữ nguyên, chỉ đổi `model=`.

OpenOBB11 giữ **lõi pretrained và head dense của OpenOBB8**. Thay đổi nằm ở **vai trò và cách train của tầng suy luận quan hệ**:
- Từ một tầng "chấm lại điểm" (học một-nhiều), nó trở thành **bộ giải mã tập kết quả** (học một-một).
- Mỗi object chỉ còn đúng một ứng viên dương. Mọi bản sao và mảnh vụn là âm.
- Đầu ra cuối cùng chính là đầu ra của tầng này, **không qua NMS**.

## 1. Bài học từ v3 → v10 và số đo

**Box trùng là lỗi chính nhìn thấy được** (ảnh user: một bức tường dài ra nhiều box chồng nhau).
- v10: 16.7% số detection là box trùng (F47).
- Phân tích trên 53 ảnh val với `openobb10x.pt` (`research/analysis/dup_analysis.py`; box thô trước NMS, nhiều cách lọc, thước YOLO):

| Cách lọc | mAP50 | mAP50-95 | Box trùng (conf 0.25) |
|---|---|---|---|
| NMS ProbIoU 0.7 (mặc định) | 0.675 | 0.438 | 1191 / 7355 |
| NMS 0.6 | 0.689 | 0.442 | 354 |
| NMS 0.5 | 0.693 | 0.444 | 115 |
| NMS 0.3 | 0.694 | 0.446 | 66 |
| NMS 0.7 + bỏ box nằm ≥ 80% trong box điểm cao hơn | 0.672 | 0.437 | 1063 |

- Các cặp box cùng class còn chồng nhau sau NMS: **2587 chồng lệch**, **738 box ngắn nằm trong box dài**, 510 gần trùng khít.
  - Với vật mảnh, IoU gần như chỉ còn là tỉ lệ độ dài. Một bản sao 200 px của bức tường 300 px có IoU 0.67, nên qua được NMS 0.7.
  - Lệch ngang 2–3 px với tường dày 6 px cũng làm IoU tụt mạnh.

**Vì sao có nhiều bản sao: ba nguyên nhân ở kiến trúc.**
1. Head dense được train một-nhiều có chủ đích: `10 + chiều dài / 16` điểm dương mỗi object (tối đa 48), điểm nào cũng được dạy vẽ trọn object với điểm cao. Tường dài ra hàng chục bản sao, lệch nhau ở hai đầu.
2. NMS theo IoU mù với vật mảnh (bảng trên). Hạ ngưỡng thì bớt trùng trên data này, nhưng:
   - ngưỡng thấp xoá nhầm object thật đứng sát nhau (xe trên DOTA, tường song song sát nhau);
   - ngưỡng đúng mỗi dataset một khác;
   - ngay ở NMS 0.3 vẫn còn 66 box trùng.
3. Tầng quan hệ của v8 / v10 có đúng công cụ để khử trùng:
   - đọc từng ứng viên trong khung xoay của nó;
   - so các ứng viên qua bias attention theo hình học tương đối, nên biết hai box cùng trục và một cái nằm trong cái kia.

   Nhưng nó được **train một-nhiều**: mọi ứng viên có ProbIoU ≥ 0.5 với object đều là dương (`openobb8_loss.py`).
   Nó được dạy **giữ** bản sao, không bao giờ học bỏ chúng. Đây là lý do v8 không giảm trùng so với v3 (F43, F44).

## 2. Lập luận

| Mảnh | Bài học / số đo | Cơ chế | Kỳ vọng | Điều sẽ chứng minh sai |
|---|---|---|---|---|
| **Tầng quan hệ train một-một** (mới) | Mục 1: bản sao được dạy là dương; IoU không phân biệt được bản sao với vật mảnh | Mỗi ảnh: ghép Hungarian giữa ứng viên đã tinh chỉnh và object, chi phí −(p^α · ProbIoU^β) (α = 1, β = 6 như bộ gán nhãn dense). Một ứng viên mỗi object là dương, còn lại là âm (đích 0). Tầng quan hệ thấy cả cụm bản sao, nên học được cái nào giữ. Ý tưởng: Relation Networks (Hu 2018, module quan hệ thay NMS); DETR / YOLOv10 / YOLO26 (gán một-một để bỏ NMS) | Box trùng gần 0 mà không cần NMS; chọn box khít nhất (β = 6) nên mAP50-95 của vật dài tăng | Box trùng (conf 0.25) không xuống dưới 5% số detection |
| **Suy luận không NMS** | Bỏ NMS 4000 ứng viên và bước lan delta sang 4000 anchor (4000 × 600 ProbIoU) | K ứng viên đã tinh chỉnh là đầu ra, xếp theo điểm. `final_nms` (ProbIoU) chỉ là tuỳ chọn an toàn, tắt mặc định | Hậu xử lý rẻ hơn v8 / v10 | — |
| **Nhóm khử nhiễu riêng** | Bản rung của box nhãn (v8) luôn tốt hơn ứng viên dense, nếu chung nhóm ghép sẽ "giành" hết vị trí dương | Bản rung là nhóm riêng: mỗi dòng là dương của chính object của nó (DN-DETR). Mask attention đã tách hai nhóm | Class hiếm vẫn có mẫu dương cho tầng quan hệ | — |
| **Head dense giữ một-nhiều + cân bằng class** | v10: cân bằng class là mảnh duy nhất có tín hiệu tốt (slide_door +.12, recall wall_300 +.09); slot làm tăng box trùng; lọc theo tầm với không được xác nhận (F47) | Loss dense = OpenOBB10 với `balance`, không `slots`, không `reach`. Head dense chỉ còn vai trò đề xuất | Hội tụ nhanh như v8 | — |

**Vì sao không làm kém các class đang tốt.**
- Lõi, head dense và tầng quan hệ có cùng tên tham số với v8, nên checkpoint v8 / v10 nạp thẳng được.
- Logit của tầng quan hệ = logit dense + phần sửa (khởi tạo 0). Lúc bắt đầu, điểm của ứng viên đúng bằng điểm dense.
- Loss một-một chỉ thêm nhãn âm cho bản sao; box dương vẫn học như v8.
- Rủi ro: lúc mới chuyển từ checkpoint v8 / v10, chưa có NMS nên vài epoch đầu sẽ còn nhiều trùng, cho tới khi tầng quan hệ học xong.

**Giới hạn.**
- Kiểu một-một học chậm hơn trên data nhỏ (bài học DETR ở v1 / v4).
- Ở đây nó chỉ áp lên tầng quan hệ; head dense vẫn một-nhiều, và tầng quan hệ bắt đầu từ điểm dense, nên rủi ro thấp hơn.
- K = 600 ứng viên: object thứ 601 trở đi (theo điểm) không ra được. Ảnh rất đông vật cần `rel_k` lớn hơn.

### Tham số suy luận do train quyết định (không chỉnh tay)

Hai tham số trước đây cố định trong code nay được **hiệu chỉnh trên val khi train xong**, rồi ghi vào `best.pt`
(`args.nms_iou`, `args.infer_k`). `predict` / `val` dùng chúng làm mặc định; truyền `iou=` khi predict thì ghi đè.

| Tham số | Thử | Chọn | Tác dụng |
|---|---|---|---|
| `nms_iou` (lọc trùng) | v11: không NMS, 0.7, 0.5. Model khác: 0.7, 0.6, 0.5 | fitness cao nhất (0.1·mAP50 + 0.9·mAP50-95, như YOLO); hoà thì ưu tiên không NMS | độ chính xác, không phải đoán ngưỡng theo dataset |
| `infer_k` (số ứng viên của tầng quan hệ) | K train (600), K/2, K/4 (≥ 64) | K nhỏ nhất mà fitness không giảm quá 0.002 | tốc độ: chi phí tầng quan hệ tỉ lệ với K (lớp quan hệ tỉ lệ K²) |

- Khi train: `nms_iou=auto` (mặc định), hoặc `iou=0.5` / `nms_iou=none` để cố định.
- Áp dụng cho mọi model có NMS (v3–v11). `infer_k` áp dụng cho các model có tầng quan hệ (v8, v10, v11).
- Ngưỡng conf theo class (F2) vốn đã được hiệu chỉnh theo cách này từ trước (`conf="auto"` khi predict).

### Các tối ưu khác của v11

| Tối ưu | Tốc độ | Độ chính xác |
|---|---|---|
| Không NMS, không lan delta sang 4000 anchor | bỏ ma trận ProbIoU 4000 × 600 và NMS 4000 ứng viên | — |
| Không ghép đa giác sau mạng khi `nms_iou` = không NMS | bỏ NMS đa giác trên CPU | — |
| Head dense 1 slot (bỏ slot của v10) | ít hơn 3 lần số anchor qua decode / chọn ứng viên | bớt bản sao (slot làm tăng trùng, F47) |
| `infer_k` hiệu chỉnh | tầng quan hệ nhỏ lại khi ảnh ít vật | giữ nguyên (ràng buộc 0.002) |
| Cân bằng class trong loss dense | — | class hiếm (F47: slide_door +.12) |
| Ghép một-một chọn box khít nhất (β = 6) | — | mAP50-95 của vật dài |

## 3. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Điều kiện: đi tiếp 50 epoch từ best của v10 (0.693 / 0.456; thước YOLO, val 53 ảnh).

| | v10 | v11 kỳ vọng |
|---|---|---|
| mAP50 / mAP50-95 | 0.693 / 0.456 | **0.70–0.73 / 0.46–0.49** |
| Box trùng (conf 0.25) | 16.7% detection | **< 5%** |
| wall / junction mAP50-95 | .548 / .517 | tăng (box khít nhất được chọn) |
| Class hiếm | slide_door .326, double_door .493, wall_300 .649 | giữ hoặc tăng nhẹ (cân bằng class giữ nguyên) |

- Phần tăng mAP chỉ ở mức vừa: trên data này, bỏ bản sao bằng NMS 0.5 chỉ cho +0.018 mAP50.
- Lợi ích chính là bỏ trùng mà không phải chỉnh ngưỡng NMS theo từng dataset, và bỏ được NMS.
- **Bác bỏ** nếu box trùng không xuống dưới 5%, hoặc mAP50 thấp hơn v10 quá 0.02 (< 0.673).

## 4. Train

```bash
# khuyến nghị: đi tiếp từ OpenOBB10 (hoặc OpenOBB8) đã train
openobb train model=openobb11x weights=<best.pt của v10> data=/content/data.yaml imgsz=1280 epochs=50

# hoặc từ COCO
openobb train model=openobb11x data=/content/data.yaml imgsz=1280 epochs=100
```

- Công thức như v8: lr 5e-4, backbone ×0.1, `rel_k=600`.
- Khi val, xem dòng `Duplicate boxes`, bảng theo class và ô wall_300 → junction.
- `v7_off=` / `v8_off=` dùng như v8 (trừ `refine`: v11 luôn cần tầng quan hệ).

- v11 không có ngưỡng NMS nào để chỉnh: việc chọn một box mỗi object nằm trong kiến trúc (tầng quan hệ học một-một).
  Với các model v10 đã train (chưa có cơ chế này), tạm thời có thể dùng `iou=0.5` khi predict;
  trên data này, cách đó bỏ khoảng 90% box trùng (bảng mục 1).

## 5. Tỉ lệ tận dụng

| Loại | Phần | Tỉ lệ |
|---|---|---|
| Weights pretrained | HGNetv2 + encoder lai (D-FINE COCO, Apache-2.0) | 81% tham số (như v8) |
| Code OpenOBB dùng lại | Lõi, head dense, tầng quan hệ của v8; loss dense của v10 | toàn bộ |
| Code chép từ nơi khác | không (Hungarian: `scipy.optimize.linear_sum_assignment`, BSD) | 0% |
| Ý tưởng từ công bố | Relation Networks (khử trùng bằng quan hệ), DETR / YOLOv10 / YOLO26 (một-một, không NMS), DN-DETR (nhóm khử nhiễu) | tự viết |
| Thiết kế mới | Bộ giải mã tập một-một trên đề xuất dense, đọc ứng viên trong khung xoay và so theo hình học tương đối, cho OBB vật mảnh | loss + suy luận |

Tham số và FLOPs bằng v8-x (66.58M; khoảng 969 GFLOPs ở 1280), ít hơn v10 (không có slot).
