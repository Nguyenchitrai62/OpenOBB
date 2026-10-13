# OpenOBB12 / OpenOBB13: head dense kép (một-nhiều để học, một-một để suy luận), không NMS (2026-10-13)

- `model=openobb12s | m | l | x`: lõi OpenOBB7 (HGNetv2 + LSK + hybrid encoder, pretrained COCO) + head kép.
- `model=openobb13s | m | l | x`: lõi nhanh OpenOBB9 (conv 3×3 tái tham số hoá) + cùng head kép. Lõi được chưng cất
  từ một OpenOBB đã train trên bản vẽ (`pretrain_teacher=<openobb10x.pt>`).

CLI và API giữ nguyên, chỉ đổi `model=`. Hai bản dùng chung một cơ chế, khác lõi: v12 kiểm tra cơ chế khử trùng trên
lõi đã chứng minh; v13 kiểm tra lõi nhanh với thầy cùng miền.

## 1. Bài học từ v3 → v11

| Bài học | Số đo |
|---|---|
| Độ chính xác đến từ lõi pretrained + head dense một-nhiều | v3 / v8 / v10: 0.69–0.72 mAP50 (thước YOLO). Thêm mọi tầng phía trên (ngữ cảnh, slot, tầng quan hệ) không tăng độ chính xác: F43, F44, F47, F50 |
| Tầng quan hệ chiếm phần lớn thời gian suy luận | v11 13.6 ms so với lõi dense không tầng quan hệ 5.4 ms (G4, batch 16, F50) |
| Box trùng sinh ra từ chính head một-nhiều (tới 48 điểm dương mỗi object) | v10: 16.7% detection là box trùng; NMS IoU mù với vật mảnh (F49) |
| Một-một bằng Hungarian trên tầng quan hệ học chậm trên 250 ảnh | v11: recall tụt (wall → background 21% so với 14%), vẫn phải chọn NMS 0.5 (F50) |
| Lõi nhanh v9 thiếu tiên nghiệm; thầy COCO không đủ | v9 + chưng cất COCO 0.539, T2 không đạt (P3 0.813, P4 0.787) (F50) |

## 2. Lập luận

| Mảnh | Cơ chế | Kỳ vọng | Điều sẽ chứng minh sai |
|---|---|---|---|
| **Head một-một dạng dense** (mới, cả v12 và v13) | Head thứ hai cùng thiết kế với head dense, cùng đặc trưng. Gán nhãn cùng thước đo (điểm^1 × ProbIoU^6) nhưng chỉ **top-1 mỗi object**; mọi điểm khác của object là âm. Ý tưởng: gán kép nhất quán của YOLOv10 / YOLO26 (head một-nhiều chỉ để học, head một-một để suy luận) | Một box mỗi object mà không cần NMS. Học nhanh như head dense vì bài toán vẫn là "điểm nào khớp object nhất" trên lưới dense, không cần decoder | Box trùng (conf 0.25) ≥ 3% khi không NMS |
| **Bỏ tầng quan hệ** | Bốn lần đo không thấy tầng quan hệ tăng độ chính xác (mục 1) | Suy luận nhanh hơn v10 / v11 rõ rệt; độ chính xác ngang v10 | mAP50 thấp hơn v10 quá 0.02 |
| **Head một-nhiều giữ nguyên** | Loss dense của OpenOBB10 có cân bằng class (mảnh duy nhất có tín hiệu tốt ở v10, F47), không slot (slot làm tăng trùng), không `reach` | Đặc trưng được học như v10; head một-một hưởng đặc trưng đó qua lõi chung | — |
| **Khởi động head một-một** | Khi nạp checkpoint một head (v7 / v8 / v10), head một-một được chép từ head một-nhiều (`copy_head_to_o2o`) | Ngay từ đầu, điểm của head một-một bằng điểm dense; loss một-một chỉ thêm nhãn âm cho bản sao | — |
| **NMS chỉ còn là dự phòng hiệu chỉnh** | Cuối train thử: không NMS, 0.7, 0.5 trên val; lưu vào `best.pt` (như 0.9.1) | Bước hiệu chỉnh chọn "không NMS" | Hiệu chỉnh chọn NMS và fitness hơn "không NMS" quá 0.005 |
| **v13: thầy cùng miền** | Thầy "dfine" là lõi của OpenOBB10 đã train trên bản vẽ (cả LSK, ScaleGuard, nhánh stride 4), không phải D-FINE COCO. Chưng cất trên ảnh đa dạng (COCO + bản vẽ): ghép hàm số trên đầu vào rộng (Beyer 2022, "patient and consistent teacher"). DINOv2 giữ vai trò thầy ngữ nghĩa | T2 đạt (cosine giữ lại P3, P4 ≥ 0.8) vì thầy và trò cùng cấu trúc tầng P3–P5 trên cùng miền | T2 < 0.8 |

**Vì sao không làm kém các class đang tốt.**
- Head một-nhiều, loss và lõi giống v10 bỏ slot. Head một-một là thêm vào, không thay phần nào đã chạy tốt.
- Gradient của head một-một cũng đi vào lõi chung. Trọng số `w_o2o = 1`: hai head cùng mức, như YOLOv10.

**Giới hạn.**
- Hai object khác nhau rơi vào cùng một điểm lưới thì chỉ một được ra (head một-một). Với vật nhỏ sát nhau ở P3 (stride 8) đây là rủi ro: theo dõi recall của junction.
- v13 phụ thuộc vào thầy: lõi trò không thể tốt hơn thầy nhiều trên miền của thầy.

## 3. Chi tiết từng phần và tỉ lệ tận dụng

| Phần | v12-x | v13-x | Nguồn |
|---|---|---|---|
| Backbone | HGNetv2 33.2M | lõi tái tham số hoá 16.2M | v12: D-FINE (Apache-2.0), init COCO. v13: tự viết (OpenOBB9) |
| LSK + encoder / neck | LSK 8.5M + hybrid encoder 20.7M + nhánh stride 4 0.3M | neck 9.4M | v12: encoder D-FINE (Apache) + LSK tự viết. v13: tự viết |
| Head một-nhiều | 2.6M | 2.6M | tự viết (head dense dị hướng của v3 / v7) |
| Head một-một | 2.6M | 2.6M | tự viết (mới) |
| Loss | `OpenOBB12Loss` = loss dense OpenOBB10 × 2 cách gán | như v12 | tự viết |
| **Tổng** | **68.0M** | **30.9M** | |

Tận dụng phần có sẵn (tính theo tham số):
- v12: khoảng 79% (backbone + encoder D-FINE). Phần mới tự viết: head một-một, loss, cách suy luận.
- v13: 0% code / weights có sẵn. Tiên nghiệm lấy từ chưng cất (DINOv2 Apache-2.0 + OpenOBB10 của user).

## 4. Chỉ số kỳ vọng (ghi TRƯỚC khi train)

Điều kiện: Wall_Color, val 53 ảnh, thước YOLO, G4, 100 epoch, cùng `imgsz` với v10. Mốc: v10 0.693 / 0.456, YOLO26x 0.746 / 0.499.

| | v10 | **v12 kỳ vọng** | **v13 kỳ vọng** |
|---|---|---|---|
| mAP50 / mAP50-95 | 0.693 / 0.456 | 0.68–0.71 / 0.45–0.47 | 0.64–0.68 / 0.41–0.45 |
| Box trùng (conf 0.25), không NMS | 16.7% (có NMS 0.7) | < 3% | < 3% |
| NMS hiệu chỉnh chọn | 0.5 | không NMS (hoặc chênh fitness ≤ 0.005) | như v12 |
| Suy luận (G4, batch 16) | — | nhanh hơn v11 (13.6 ms) ít nhất 30% | ≤ 6 ms, hậu xử lý gần 0 |
| T2 chưng cất (P3 / P4) | — | — | ≥ 0.8 / ≥ 0.8 |

- **v12 bị bác bỏ** nếu mAP50 < 0.673, hoặc box trùng không NMS ≥ 3%.
- **v13 bị bác bỏ** nếu T2 không đạt, hoặc mAP50 < 0.62 (thấp hơn v9 hiện tại 0.539 không đủ để coi là lõi nhanh dùng được).
- Nếu v12 đạt mà v13 không: cơ chế head kép đúng, lõi nhanh cần thầy / dữ liệu khác (`pretrain=dota`).

## 5. Train (Colab)

v12, từ COCO (so cùng điều kiện với v10):

```bash
openobb train model=openobb12x data=/content/data.yaml imgsz=1280 epochs=100
```

v13, chưng cất lõi từ OpenOBB10 đã train rồi train detector (một lệnh):

```bash
openobb train model=openobb13x data=/content/data.yaml imgsz=1280 epochs=100 pretrain=coco pretrain_teacher=/content/drive/MyDrive/openobb10x.pt pretrain_dir=/content/drive/MyDrive/openobb_pretrain
```

- `pretrain_teacher` nhận bất kỳ checkpoint OpenOBB đã train có lõi dense (v3, v5, v7, v8, v10, v11, v12).
- Bước chưng cất mất khoảng 45 phút G4 (20 epoch, COCO val + ảnh train). Kết quả được cache trong `pretrain_dir`, lần sau dùng lại.
- Log chưng cất in cosine giữ lại P3 / P4 (cổng T2). Gửi kèm log này.
- Khi val, xem dòng `Duplicate boxes`, cột `o2o_loss` trong `results.csv`, và dòng hiệu chỉnh NMS cuối train.
- Có thể đi tiếp từ checkpoint một head: `weights=<best.pt của v7 / v8 / v12>`; head một-một được chép từ head một-nhiều.
