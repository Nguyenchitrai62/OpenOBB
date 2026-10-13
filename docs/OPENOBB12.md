# OpenOBB12: head dense kép (một-nhiều để học, một-một để suy luận), không NMS (2026-10-13)

- `model=openobb12s | m | l | x`: lõi OpenOBB7 (HGNetv2 + LSK + hybrid encoder, pretrained COCO) + head kép.

CLI và API giữ nguyên, chỉ đổi `model=`. v12 kiểm tra cơ chế khử trùng trên lõi đã chứng minh.
OpenOBB13 (bản nhắm độ chính xác: luồng độ phân giải gốc + lưới P2, xây trên v12): [OPENOBB13.md](OPENOBB13.md).
v13 của 0.10.0 (lõi nhanh v9 + head kép) đã bị thay vì chỉ nhắm tốc độ.

## 1. Bài học từ v3 → v11

| Bài học | Số đo |
|---|---|
| Độ chính xác đến từ lõi pretrained + head dense một-nhiều | v3 / v8 / v10: 0.69–0.72 mAP50 (thước YOLO). Thêm mọi tầng phía trên (ngữ cảnh, slot, tầng quan hệ) không tăng độ chính xác: F43, F44, F47, F50 |
| Tầng quan hệ chiếm phần lớn thời gian suy luận | v11 13.6 ms so với lõi dense không tầng quan hệ 5.4 ms (G4, batch 16, F50) |
| Box trùng sinh ra từ chính head một-nhiều (tới 48 điểm dương mỗi object) | v10: 16.7% detection là box trùng; NMS IoU mù với vật mảnh (F49) |
| Một-một bằng Hungarian trên tầng quan hệ học chậm trên 250 ảnh | v11: recall tụt (wall → background 21% so với 14%), vẫn phải chọn NMS 0.5 (F50) |

## 2. Lập luận

| Mảnh | Cơ chế | Kỳ vọng | Điều sẽ chứng minh sai |
|---|---|---|---|
| **Head một-một dạng dense** (mới) | Head thứ hai cùng thiết kế với head dense, cùng đặc trưng. Gán nhãn cùng thước đo (điểm^1 × ProbIoU^6) nhưng chỉ **top-1 mỗi object**; mọi điểm khác của object là âm. Ý tưởng: gán kép nhất quán của YOLOv10 / YOLO26 (head một-nhiều chỉ để học, head một-một để suy luận) | Một box mỗi object mà không cần NMS. Học nhanh như head dense vì bài toán vẫn là "điểm nào khớp object nhất" trên lưới dense, không cần decoder | Box trùng (conf 0.25) ≥ 3% khi không NMS |
| **Bỏ tầng quan hệ** | Bốn lần đo không thấy tầng quan hệ tăng độ chính xác (mục 1) | Suy luận nhanh hơn v10 / v11 rõ rệt; độ chính xác ngang v10 | mAP50 thấp hơn v10 quá 0.02 |
| **Head một-nhiều giữ nguyên** | Loss dense của OpenOBB10 có cân bằng class (mảnh duy nhất có tín hiệu tốt ở v10, F47), không slot (slot làm tăng trùng), không `reach` | Đặc trưng được học như v10; head một-một hưởng đặc trưng đó qua lõi chung | — |
| **Khởi động head một-một** | Khi nạp checkpoint một head (v7 / v8 / v10), head một-một được chép từ head một-nhiều (`copy_head_to_o2o`) | Ngay từ đầu, điểm của head một-một bằng điểm dense; loss một-một chỉ thêm nhãn âm cho bản sao | — |
| **NMS chỉ còn là dự phòng hiệu chỉnh** | Cuối train thử: không NMS, 0.7, 0.5 trên val; lưu vào `best.pt` (như 0.9.1) | Bước hiệu chỉnh chọn "không NMS" | Hiệu chỉnh chọn NMS và fitness hơn "không NMS" quá 0.005 |

**Vì sao không làm kém các class đang tốt.**
- Head một-nhiều, loss và lõi giống v10 bỏ slot. Head một-một là thêm vào, không thay phần nào đã chạy tốt.
- Gradient của head một-một cũng đi vào lõi chung. Trọng số `w_o2o = 1`: hai head cùng mức, như YOLOv10.

**Giới hạn.**
- Hai object khác nhau rơi vào cùng một điểm lưới thì chỉ một được ra (head một-một). Với vật nhỏ sát nhau ở P3 (stride 8) đây là rủi ro: theo dõi recall của junction.

## 3. Chi tiết từng phần và tỉ lệ tận dụng

| Phần | v12-x | Nguồn |
|---|---|---|
| Backbone | HGNetv2 33.2M | D-FINE (Apache-2.0), init COCO |
| LSK + encoder | LSK 8.5M + hybrid encoder 20.7M + nhánh stride 4 0.3M | encoder D-FINE (Apache) + LSK tự viết |
| Head một-nhiều | 2.6M | tự viết (head dense dị hướng của v3 / v7) |
| Head một-một | 2.6M | tự viết (mới) |
| Loss | `OpenOBB12Loss` = loss dense OpenOBB10 × 2 cách gán | tự viết |
| **Tổng** | **68.0M** | |

Tận dụng phần có sẵn (tính theo tham số):
- v12: khoảng 79% (backbone + encoder D-FINE). Phần mới tự viết: head một-một, loss, cách suy luận.

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Điều kiện: Wall_Color, val 53 ảnh, thước YOLO, G4, 100 epoch, cùng `imgsz` với v10. Mốc: v10 0.693 / 0.456, YOLO26x 0.746 / 0.499.

| | v10 | **v12 kỳ vọng** |
|---|---|---|
| mAP50 / mAP50-95 | 0.693 / 0.456 | 0.68–0.71 / 0.45–0.47 |
| Box trùng (conf 0.25), không NMS | 16.7% (có NMS 0.7) | < 3% |
| NMS hiệu chỉnh chọn | 0.5 | không NMS (hoặc chênh fitness ≤ 0.005) |
| Suy luận (G4, batch 16) | — | nhanh hơn v11 (13.6 ms) ít nhất 30% |

- **v12 bị bác bỏ** nếu mAP50 < 0.673, hoặc box trùng không NMS ≥ 3%.

## 5. Train (Colab)

Từ COCO (so cùng điều kiện với v10):

```bash
openobb train model=openobb12x data=/content/data.yaml imgsz=1280 epochs=100
```

- Khi val, xem dòng `Duplicate boxes`, cột `o2o_loss` trong `results.csv`, và dòng hiệu chỉnh NMS cuối train.
- Có thể đi tiếp từ checkpoint một head: `weights=<best.pt của v7 / v8 / v12>`; head một-một được chép từ head một-nhiều.

## 6. Cập nhật 0.12.0 (sau run đầu, F54)

Run v12 đầu tiên của user đạt 0.722 / 0.468, cao nhất tới nay. Hai phát hiện từ phân tích checkpoint đó:

1. **Head dùng khi suy luận là một tham số hiệu chỉnh.**
   - Trên cùng checkpoint, head một-nhiều + NMS 0.5 cho 0.741 / 0.481, còn head một-một cho 0.718 / 0.470.
   - Train kèm head một-một làm lõi tốt hơn; nhưng ở 250 ảnh, head một-nhiều + NMS vẫn tốt hơn khi suy luận.
   - Cuối train, bước hiệu chỉnh thử `{một-một: không NMS / 0.7 / 0.5; một-nhiều: 0.7 / 0.5}` trên val, lưu `args["infer_head"]` và `args["nms_iou"]` vào `best.pt`.
   - Checkpoint đã train: `openobb calibrate model=best.pt data=data.yaml [out=calibrated.pt]`.
2. **`box=direct`: hồi quy box kiểu YOLO26.**
   - So với YOLO26x, box của v12:
     - dày hơn thật một cách hệ thống (+0.66 px trên tường 5.6 px);
     - sai số độ dày gấp đôi (1.00 so với 0.54 px);
     - đầu box sai gần gấp đôi (2.4 so với 1.3 px).
   - Nguyên nhân: độ dày và độ dài đọc từ phân phối DFL theo bin thô (stride / 2, stride). Sửa cách giải mã chỉ hết lệch, không giảm sai số (F54).
   - `box=direct` thay ba phân phối (hai đầu, độ dày) bằng ba giá trị liên tục (softplus × stride). Loss là L1 theo đơn vị stride + ProbIoU; góc và lệch ngang giữ nguyên.
   - Kỳ vọng ghi trước ở LEDGER F55.

```bash
openobb train model=openobb12x data=/content/data.yaml imgsz=1280 epochs=100 box=direct
```
