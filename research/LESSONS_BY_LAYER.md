# Bài học cốt lõi theo từng lớp của model (2026-10-12)

Tổng hợp từ mọi version (vòng cũ CAD-VLMDet, VRDet/OpenOBB1 → OpenOBB8), đối chiếu với D-FINE và các kiến trúc SOTA.
Mọi kết luận đều dẫn về số F trong [LEDGER.md](LEDGER.md). Phần nào chỉ là suy luận, chưa đo, thì ghi rõ.

## 0. Bảng tổng kết các version

Số liệu trên Wall_Color (250 train / 53 val), thước YOLO trừ khi ghi "DOTA". Mốc: YOLO26x **0.746 / 0.499**.

| Version | Lõi | Khởi tạo | Đầu ra | Kết quả tốt nhất | Kết luận |
|---|---|---|---|---|---|
| CAD-VLMDet (vòng cũ) | ConvNeXt + transformer P5, tự viết hết | từ đầu | dense OBB tự chế | AP50 0.144 (floorplan) | thất bại: từ đầu + assigner/loss chưa kiểm |
| v1 (VRDet) | D-FINE: HGNetv2 + encoder lai + decoder DETR | COCO | decoder một-một | DOTA 0.63 / 0.35 (**chưa đo thước YOLO**) | dùng được; chậm; class hiếm yếu |
| v2 | tự viết, P2–P5 | từ đầu | dense, hồi quy vị trí × log(dài) | DOTA 0.356 / 0.133, loss phẳng | thất bại: underfit do cách biểu diễn box + không pretrain |
| **v3** | HGNetv2 + encoder lai (D-FINE COCO) + LSK | COCO | dense hướng, DFL dị hướng | **0.720 / 0.474** (sau 3 vòng) | thành công; thua chủ yếu ở wall_300 |
| v4 | DINOv2 + adapter + encoder/decoder D-FINE + dense | DINOv2 + COCO | union | 0.663 / 0.457 (1 vòng), 38 ms | một phần: giúp wall_300 + độ khít, hại class hiếm, chậm |
| v5 | v3 + phân loại theo hình học + chấm lại quan hệ | từ v3 vòng 1 | dense | 0.671 / 0.448 | không cải thiện (quan hệ đọc 1 điểm, vị trí tuyệt đối) |
| v6 | backbone tự thiết kế (depthwise, dải) | từ đầu | dense (head v3) | chững 0.55 / 0.32 sau 150 ep | thất bại: không có tiên nghiệm; ít FLOPs mà GPU không nhanh hơn |
| v7 | v3 + 4 sửa lỗi F37 | COCO | dense | không train riêng | nằm trọn trong v8 |
| **v8** | lõi v7 + tầng suy luận quan hệ theo hướng | COCO | dense + tinh chỉnh 600 ứng viên | **đang chạy: ep37 0.607 / 0.375** (F42) | hội tụ nhanh nhất từ trước tới nay; chờ số cuối |

**Quy luật lớn nhất:** mọi bản thành công đều đứng trên **trọng số pretrained của D-FINE COCO** (backbone + encoder, khoảng 81% tham số).
Mọi bản train từ đầu đều thất bại, bất kể kiến trúc tốt hay dở. Phần "D-FINE" thật sự mang lại thành công là **trọng số**, không phải
decoder hay các mẹo loss của nó: v3 bỏ hẳn decoder mà vẫn là bản tốt nhất (theo thước YOLO).

---

## 1. Lớp khởi tạo / tiên nghiệm: quyết định nhiều nhất

| Bằng chứng | Số |
|---|---|
| Cùng data, cùng 50 epoch: COCO-init (v3) so với từ đầu (v2, v6) | khoảng gấp đôi mAP50-95 (F39) |
| Từ đầu, train tới 150 epoch (v6) | chững 0.55 / 0.32; loss train vẫn giảm, val đứng (F40) → giới hạn là **tổng quát hoá**, không phải dung lượng |
| Pretrain cùng miền (checkpoint FloorPlanCAD) so với dung tích lớn hơn (X từ COCO) | S-từ-CAD 0.40 ngay đầu; X-từ-COCO mới 0.27 ở ep16 (F21) |
| Objects365 → COCO thay vì COCO | −2.05 trên DOTA (F15): tiên nghiệm "nhiều hơn" không tự động tốt hơn |
| DINOv2 thêm vào (v4) | wall_300 +0.08 / +0.11, độ khít tường +0.027; class hiếm tệ hơn (F33) |

**Bài học.**
1. Với vài trăm ảnh, **tiên nghiệm chính là model**. Một kiến trúc mới không có tiên nghiệm sẽ thua một kiến trúc cũ có tiên nghiệm khoảng 0.15 mAP.
2. Phần mới phải **đặt lên trên** lõi pretrained và **bắt đầu bằng phép đồng nhất** (đầu ra mới = 0), như v7/v8. Khi đó bản mới có cận dưới là bản cũ, và so sánh sạch.
3. Muốn lõi hoàn toàn của mình thì phải **tự tạo tiên nghiệm** (chưng cất từ thầy, v9), không thể train từ đầu.
4. Các nguồn tiên nghiệm bổ sung nhau: COCO/HGNet giỏi class hiếm và cửa, DINOv2 giỏi vật tí hon và độ khít (F33). Đây là cơ sở của chưng cất hai thầy (v9).

## 2. Lớp backbone

| Bằng chứng | Số |
|---|---|
| P5 của HGNetv2 chết ở cả 3 checkpoint v3: độ lớn 3566, đặt = 0 không đổi detection | khoảng 75 GFLOPs (8%) vô ích (F37) |
| Nguyên nhân | BN đóng băng + lr 1e-3 + LSK nhân không chặn `x·(1+m)` khuếch đại lên 10¹⁰ |
| Cùng nhóm BN đó đã nổ ở v1 (F22) và v3 vòng 1 (F31) | 3 lần, cùng một lớp `encoder.input_proj.2` |
| v6: 504 GFLOPs, 25.7 ms; v3: 909 GFLOPs, 23.1 ms | FLOPs ≠ độ trễ; depthwise / BN rời bị nghẽn băng thông bộ nhớ (F39) |
| Stem + max-pool của HGNet giữ nét mảnh: dịch ảnh 1–7 px, tâm lệch 0.2–0.5 px | định vị không phải nút thắt (F37) |
| YOLO26x dồn 13% FLOPs vào stride 4, chỉ 10% ở 1/32 (768 kênh); HGNet-B5 có 2048 kênh ở 1/32 | phân bổ kiểu mạng phân loại ngược với nhu cầu vật nhỏ |

**Bài học.**
1. **Ổn định số học là yêu cầu kiến trúc, không phải chuyện chỉnh LR.** Không cổng nhân không chặn; BN phải được train (gộp lúc suy luận) hoặc đầu ra phải được chặn độ lớn.
   Một tầng "chết" không báo lỗi gì: mAP vẫn ổn, chỉ là tầng đó không đóng góp. Phải **đo** (đặt tầng = 0, xem detection) mới thấy.
2. **Tính toán ưu tiên độ phân giải cao**; 1/32 chỉ cần mỏng.
3. **Đo độ trễ, không đếm FLOPs.** Trên GPU, conv 3×3 dày (cường độ tính khoảng 1000 FLOP/byte) nhanh hơn nhiều so với depthwise (khoảng 5 FLOP/byte) cùng FLOPs.
4. Giữ những gì đã chứng minh giữ nét: nhánh max-pool khi hạ độ phân giải, stem stride 4 không làm hỏng nét 4 px.

## 3. Lớp neck / encoder / ngữ cảnh

| Bằng chứng | Số |
|---|---|
| AIFI (attention toàn ảnh ở 1/32) trên CAD | đầu vào hằng số, attention gần đều (682 so với 667 px nếu đều tuyệt đối) (F37) |
| Thumbnail ngữ cảnh toàn ảnh (H6) | −1.93 rồi −3.88 trên DOTA; học scene prior, HC −20 đến −27 (F6) |
| Ngữ cảnh dải sau encoder (H3) | −0.13 (DOTA) |
| LSK (kernel lớn trước encoder) trên FloorPlanCAD | **+1.33 / +2.24** (c6), nhưng cổng không chọn theo nội dung (F37) |
| Thêm tầng P2 cho decoder (c5) | **−9.1 / −2.8** (FPC) |
| Bán kính bằng chứng của logit class | nửa trong khoảng 40 px, 90% trong 120–186 px; 75–93% nằm ngoài box (F37) |
| Khối FPN P3 của encoder | 23% FLOPs cả model, khối nặng nhất |

**Bài học.**
1. Ngữ cảnh có ích là ngữ cảnh **tầm trung, cục bộ** (vài chục tới khoảng 200 px), **ở 1/8–1/16**. Ngữ cảnh toàn ảnh hoặc không có tác dụng (AIFI) hoặc gây hại (thumbnail: model học "cảnh này thường có gì" thay vì nhìn vật).
2. Kernel lớn có ích vì **mở rộng vùng nhìn**, không vì "chọn kernel". Giữ phần depthwise lớn, bỏ cổng chọn.
3. Muốn chi tiết stride 4: **bơm vào P3** (space-to-depth, khởi tạo 0) hoặc **lấy mẫu stride 4 trong tầng thứ hai** (v8). Thêm hẳn một tầng P2 cho query/head làm loãng ứng viên và tụt mạnh (c5).
4. Thiết kế của RT-DETR (attention chỉ ở 1/32, vì COCO có vật lớn) **không chuyển được** sang data vật nhỏ.

## 4. Lớp head: biểu diễn box

| Bằng chứng | Số |
|---|---|
| v2: hồi quy "vị trí dọc trục × log(chiều dài)" + L1 đầu mút | `end_loss` 1.20 → 0.91 rồi đứng yên 66 epoch; tỉ lệ khít kẹt 0.35 (F25, F27) |
| v3: hai khoảng cách tới đầu mút dạng DFL (bin = stride) + độ dày DFL (bin = stride/2) | box 1.53 → 0.51, dfl 1.67 → 0.67 trong 26 epoch; tỉ lệ khít 0.66 (F28, F31) |
| DFL của v3 không bão hoà, độ dày một đỉnh, bất định khoảng 2.7 px | phần box làm tốt (F37) |
| Loss Gaussian (KLD, ProbIoU) mù góc với box gần vuông | cần thêm loss góc kiểu sin²(2Δθ) hoặc L1 góc đã căn ±90° (F14) |
| Lỗi `probiou` không bất biến tỉ lệ (= 1 với box < 100 px) | vô hiệu cả H4 / H4c (F10) |

**Bài học.**
1. **Mỗi cạnh một phân phối rời rạc có giới hạn** (DFL / FDR của D-FINE) học được; tham số hoá toàn cục (tâm + log độ dài) cho vật dài thì không. Lý do: phân phối cho gradient cục bộ, có giới hạn ở từng cạnh; log-dài × vị trí ghép hai biến làm mặt loss xấu.
2. Đơn vị bin phải theo **hướng của vật**: dọc trục theo stride, ngang trục mịn hơn (stride/2). Vật mảnh cần độ phân giải ngang cao hơn dọc.
3. Mọi phép đo hình học xoay (ProbIoU, IoU đa giác) phải có **test bất biến tỉ lệ / xoay** trước khi dùng làm loss. Một lỗi ở đây vô hiệu hoá cả loạt thí nghiệm.

## 5. Lớp head: phân loại (nút thắt thật)

| Bằng chứng | Số |
|---|---|
| wall_300 → junction | 32% (v3, F35), 49% (v6, F39) |
| wall_300 cách junction gần nhất | 3.9 px; khác nhau ở **cách sắp xếp**, không ở hình dạng (F38) |
| 43% wall_300 có biên top1 − top2 < 0.2 | phân loại lưỡng lự đúng chỗ thua YOLO (F37) |
| Head class: v3 5% FLOPs, YOLO26x 20% | head quá nhẹ so với độ khó |
| v5: cộng 7 số đo hình học vào nhánh class + chấm lại quan hệ đọc **1 điểm**, vị trí **tuyệt đối** | không cải thiện (F35) |
| F32: riêng wall_300 chiếm khoảng 95% khoảng cách mAP50 của v3 với YOLO26x | khoảng cách là ngữ nghĩa, không phải định vị |

**Bài học.**
1. Với vật tí hon trông giống nhau, **class được quyết định bởi hàng xóm và cách sắp xếp**, không bởi điểm ảnh của chính vật.
2. Đọc đặc trưng tại một điểm (dù có thêm số đo hình học) **không đủ**. Cần đọc một vùng có cấu trúc quanh vật, trong **khung của chính vật** (v8: lưới 8×3 + vòng ngữ cảnh).
3. Quan hệ giữa các ứng viên phải mã hoá bằng **hình học tương đối** (box j trong khung box i), bất biến khi xoay. Vị trí tuyệt đối (v5) không mang thông tin "sắp xếp".
4. Head class phải đủ nặng; YOLO dành 20% FLOPs cho head là có lý do.

## 6. Lớp gán nhãn / số mẫu dương

| Bằng chứng | Số |
|---|---|
| Decoder một-một (v1, v4) với class hiếm | class hiếm yếu ở cả DOTA (E1b: HC, SBF, RA) và Wall_Color (v4: slide_door .146 so với v3 .273) (F3, F33) |
| RFS (lấy mẫu lại class hiếm) | **+2.45** DOTA (F3), mặc định từ đó |
| Head dense (TAL một-nhiều) so với decoder | dense giỏi vật nhỏ, decoder giỏi vật lớn / cần ngữ cảnh; hợp hai bên +0.66 (F16) |
| DETR cần khoảng 2000+ bước; head dense pretrained học nhanh | v3 plateau sau khoảng 30 epoch fine-tune (F19, F31) |
| YOLO26: STAL nới box < 16 px lên 16 px khi chọn ứng viên; ProgLoss một-nhiều → một-một | vật tí hon luôn có đủ điểm dương (F12) |

**Bài học.**
1. **Học bằng một-nhiều, quyết định bằng tập hợp.** Một-nhiều (TAL) cho nhiều mẫu dương mỗi vật nên học nhanh, kể cả class hiếm. Bước "chọn một box cho mỗi vật" và suy luận quan hệ nên đặt ở tầng sau, trên các ứng viên đã tốt.
2. Đây chính là điều SOTA đã hội tụ về: Co-DETR / DDQ (dense đề xuất → sparse suy luận), YOLO26 (hai head + ProgLoss), H-DETR (nhóm query một-nhiều). v8 là phiên bản hướng-OBB của cùng nguyên lý.
3. Class hiếm cần **tín hiệu học riêng**: RFS ở mức data; ở v8 là bản rung của box nhãn (tầng hai được dạy cả những vật mà head dense bỏ sót).

## 7. Lớp suy luận thứ hai (quan hệ / tinh chỉnh)

| Version | Cách làm | Kết quả |
|---|---|---|
| v1 (decoder DETR) | query đọc mẫu biến dạng + self-attention giữa query, một-một | thắng vật lớn / ngữ cảnh (wardrobe +7.7, sofa +6.3 trên FPC; BD, HA trên DOTA) nhưng class hiếm yếu, học chậm |
| c2 (dense đề xuất → decoder) | query lấy từ head dense | bằng c1, chậm hơn (FPC) |
| v5 | self-attention trên 600 ứng viên, đặc trưng 1 điểm, vị trí tuyệt đối, chỉ sửa class | không giúp (F35) |
| **v8** | đọc trong khung xoay (stride 4 + vòng ngữ cảnh), bias hình học tương đối, sửa class + box, đồng nhất lúc khởi tạo, bản rung box nhãn | đang chạy; hội tụ nhanh (F42) |

**Bài học.**
1. Tầng thứ hai chỉ có ích khi **mỗi ứng viên có đặc trưng riêng đủ giàu** (lấy mẫu có cấu trúc), không phải một vector tại một điểm.
2. Quan hệ phải **tương đối và bất biến**; nếu không, model phải học lại cùng một quan hệ ở mọi vị trí / góc, data nhỏ không đủ.
3. Tầng thứ hai phải **bắt đầu bằng đồng nhất** trên một tầng một đã tốt. Decoder DETR của v1 không như vậy (học từ đầu toàn bộ quyết định), nên chậm và yếu ở class hiếm.
4. Chưa tách được: phần lợi của v8 đến từ tầng quan hệ hay từ 4 sửa lỗi lõi (v7). Cần `v8_off=refine` cùng công thức.

## 8. Lớp hậu xử lý

| Bằng chứng | Số |
|---|---|
| Trần 300 query; trang dày 150–400 vật | 600–900 query khi suy luận +0.7, latency không đổi (F4, F9) |
| Argmax một class mỗi query | −0.16 / −0.38 (F14 / e14) |
| NMS 0.1 khi chỉ 1 tile/ảnh | xoá vật kề nhau; đổi sang 0.7 (F19) |
| v8: anchor trùng ứng viên giữ điểm cũ sẽ thắng NMS | phải lan delta sang anchor trùng (OPENOBB8 §3) |

**Bài học.** Số ứng viên và ngưỡng NMS phải theo **mật độ vật** của data, không theo mặc định COCO. Mọi tầng sửa điểm sau NMS-trước phải nhất quán với NMS (lan delta), nếu không phần sửa bị vô hiệu ở bước cuối.

## 9. Lớp công thức train

| Bằng chứng | Số |
|---|---|
| lr 1e-3 với lõi pretrained (v1, v3 vòng 1) | BN P5 nổ, NaN (F22, F31). v8 đang chạy 1e-3 **có chặn độ lớn**: chưa nổ ở ep37 (F42) |
| 250 ảnh, batch 32, 30 epoch = 210 bước | mAP = 0 tới ep10; cần ≥ 2000 bước cho kiểu DETR (F19) |
| Mosaic thu nhỏ 0.5× trên CAD | −3.9 (F8); symbol bị thu nhỏ |
| Tắt mosaic ở 10 epoch cuối (v4) | box_loss 0.08 → 0.28 (đổi phân phối) mà mAP vẫn tăng (F33) |
| Nhiều vòng fine-tune nối tiếp (v3: 100 + 50 + 50) | mỗi vòng lại warmup; khó so ngân sách giữa các version (F33) |

**Bài học.** Augmentation phải **giữ độ phân giải của nét/symbol**. So version phải **cùng ngân sách bước và cùng một vòng train**, không so một bản 1 vòng với một bản 3 vòng.

## 10. Lớp đo lường (bài học về quy trình)

1. **Cùng thước.** Thước DOTA thấp hơn thước YOLO khoảng 0.13 / 0.16 trên data tường mảnh (F29, F31). Kết luận "thua xa YOLO" của F23 phần lớn là do thước.
   **Hệ quả còn treo:** v1 đạt 0.63 / 0.35 **thước DOTA**, cao hơn v3 (0.593 / 0.312 thước DOTA), nhưng v1 chưa bao giờ được đo thước YOLO.
   Thứ hạng "v3 > v1" trên Wall_Color **chưa được chứng minh**. Nếu v1 cao hơn thật, đó là bằng chứng mạnh cho tầng suy luận thứ hai (decoder) mà v8 đang thử.
2. **Đo cơ chế trước khi thiết kế lại** (F37): đặt tầng = 0, đo cổng trên nét / trên nền, đo vùng nhìn của logit. Ba phát hiện lớn nhất (P5 chết, LSK không chọn, nút thắt là phân loại) đều đến từ đây, không đến từ mAP.
3. **Một yếu tố mỗi lần** + đồng nhất lúc khởi tạo cho phép đối chứng sạch. v5 đổi checkpoint, LR và xoay cùng lúc nên không kết luận được gì (F35).
4. Confusion matrix đổi hẳn giả thuyết (F35: wall_300 không nhầm sang wall mà sang junction). Phải có trong mọi lần val.

## 11. Đối chiếu với SOTA: OpenOBB đang đứng ở đâu

| Nguyên lý SOTA | Ai làm | OpenOBB đã có? |
|---|---|---|
| Lõi pretrained mạnh | mọi SOTA | có (D-FINE COCO); v9 sẽ tự tạo bằng chưng cất |
| Phân phối cạnh (DFL / FDR) | GFL, D-FINE | có, dị hướng (v3) |
| Một-nhiều để học, một-một / tập hợp để quyết định | Co-DETR, DDQ, YOLO26, H-DETR | v8 (dense + tầng quan hệ); chưa có ProgLoss |
| STAL cho vật tí hon | YOLO26 | **chưa** (dải ứng viên đa tầng ≥ 1 stride của v3 gần giống, chưa so) |
| Head nặng (20% FLOPs) | YOLO26 | một phần (StripMix trong tháp class, v7) |
| Max-pool giữ đỉnh tín hiệu (SPPF) | YOLO | có ở stem HGNet; v9 đưa vào mọi bước hạ độ phân giải |
| Tái tham số hoá cho GPU | RepVGG, YOLOv6/7 | **chưa** (v9) |
| Quan hệ giữa object theo hình học tương đối | Relation Networks, Swin bias | **có, mới cho OBB** (v8) |
| Lấy mẫu trong khung xoay | RoI Transformer, Oriented R-CNN | có (v8), không cần two-stage đầy đủ |

## 12. Tóm tắt một dòng cho mỗi lớp

| Lớp | Bài học cốt lõi |
|---|---|
| Khởi tạo | Tiên nghiệm quyết định; phần mới đặt lên trên, bắt đầu bằng đồng nhất |
| Backbone | Chặn mọi độ lớn; tính toán ở độ phân giải cao; đo độ trễ, không đếm FLOPs |
| Neck / ngữ cảnh | Ngữ cảnh tầm trung ở 1/8–1/16; ngữ cảnh toàn ảnh vô ích hoặc có hại |
| Box | Mỗi cạnh một phân phối có giới hạn, theo hướng của vật |
| Class | Vật tí hon được phân loại bằng hàng xóm và cách sắp xếp |
| Gán nhãn | Học bằng một-nhiều, quyết định bằng tập hợp; class hiếm cần tín hiệu riêng |
| Tầng hai | Đặc trưng có cấu trúc trong khung vật + quan hệ tương đối + đồng nhất lúc đầu |
| Hậu xử lý | Theo mật độ vật; tầng sửa điểm phải nhất quán với NMS |
| Train | Giữ độ phân giải của nét; so cùng ngân sách |
| Đo | Cùng thước; đo cơ chế; một yếu tố mỗi lần; luôn có confusion matrix |
