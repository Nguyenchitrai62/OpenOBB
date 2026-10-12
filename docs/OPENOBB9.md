# OpenOBB9: lập luận thiết kế (đề xuất, CHƯA viết code, chờ user duyệt, 2026-10-12)

Theo luật của user: một version mới phải có lập luận **trước** khi code. Tài liệu này nêu từng lựa chọn thiết kế xuất phát từ bài học nào
(số F trong [research/LEDGER.md](../research/LEDGER.md)), dựa trên lý thuyết nào, kỳ vọng gì, và **phép thử nào bác bỏ được nó**.
Phép thử được xếp sao cho phép rẻ nhất chạy trước. Nếu một phép thử bị bác bỏ thì dừng, không tốn CU cho bước sau.

## 0. Tóm tắt

v9 là một kiến trúc **mới hoàn toàn và thuộc sở hữu của user** (không dùng HGNetv2 / encoder D-FINE trong model). Nó nhắm cùng lúc ba điều:

| Mục tiêu | Cách đạt | Cơ sở |
|---|---|---|
| **A. Chính xác ít nhất bằng v8** | Pretrain bằng **chưng cất từ hai thầy**: lõi D-FINE COCO (chính là lõi của v3 / v8) và DINOv2. Head giữ nguyên của v8 | Mệnh đề 1 (mục 3) + bằng chứng hai thầy bổ sung nhau (F33) |
| **B. Nhanh hơn rõ trên GPU** (nhắm ≤ 0.6× độ trễ của v8 cùng imgsz) | Lõi chỉ gồm **conv 3×3 dày tái tham số hoá** + max-pool; không depthwise, không ViT khi suy luận | Mô hình roofline (mục 4) + bài học v6 (F39) |
| **C. Không lặp lại các lỗi đã đo** | BN được train và gộp khi suy luận; không có cổng nhân không chặn; **mỗi tầng P3–P5 đều có loss chưng cất** nên không tầng nào chết được | F37 (P5 chết ở cả 3 checkpoint v3) |

Khác v6 ở ba chỗ quyết định: (1) có pretrain, và pretrain **truyền đúng biểu diễn của v3** chứ không chỉ học sơ bộ; (2) dùng conv dày, không dùng depthwise; (3) chỉ đổi lõi, giữ head của v8, nên so v9 với v8 là so đúng một yếu tố.

## 1. Bài học đã đo (đầu vào của lập luận)

| # | Bài học | Bằng chứng |
|---|---|---|
| L1 | Data nhỏ (250 ảnh) thì **kiến thức nền quyết định**. Train từ đầu thì chững | v2 kẹt ở 0.36 (thước DOTA; F25–F27); v6 khoảng 0.55 / 0.32 (F40); v3 (COCO) 0.720 / 0.474 (F31) |
| L2 | **FLOPs không phải độ trễ.** Nhiều depthwise thì GPU chậm | v6: 504 GFLOPs nhưng 25.7 ms (chưa gộp BN), v3: 909 GFLOPs, 23.1 ms (F39) |
| L3 | BN đóng băng + lr cao + cổng nhân không chặn → **tầng chết** | P5 của v3 có độ lớn 3566 → 10¹⁰; đặt P5 = 0 không đổi detection (F37) |
| L4 | **Định vị không phải nút thắt; phân loại vật tí hon dựa vào hàng xóm mới là nút thắt** | Dịch ảnh lệch < 1 px; 75–93% bằng chứng class nằm ngoài box; wall_300 → junction 32% (F35, F37) |
| L5 | **Đặc trưng DINOv2 bổ sung cho HGNet** ở đúng chỗ v3 yếu, nhưng ViT chạy khi suy luận thì chậm | v4 (DINOv2): wall_300 .683 so với .599, mAP50-95 của wall .602 so với .575; nhưng 38.1 ms (F33) |
| L6 | Chấm lại quan hệ bằng đặc trưng một điểm + vị trí tuyệt đối thì không giúp | v5 (F35); v8 đã đổi sang khung xoay + hình học tương đối |
| L7 | Head cần đủ nặng; tính toán nên ở độ phân giải cao | YOLO26x: head 20% FLOPs, stride 4 13%; v3: head 5%, 2048 kênh ở 1/32 chết (F37) |
| L8 | Pretrain rẻ phải đo được chất lượng trước khi train detection, và chi phí phải ước tính từ số đo thật | v6 chạy 17 ảnh/s ở 1280 trên G4; ước tính pretrain ban đầu của mình lệch 3–4 lần (F39) |

## 2. Thiết kế đề xuất (cỡ x, sẽ chỉnh độ rộng sau khi đo)

```
ảnh ─ Stem: conv 3×3 s2 → [conv 3×3 s2 ‖ max-pool] → 1×1                      stride 4
   ─ S4: RepCSP (1 khối)                                                         stride 4
   ─ Down(conv 3×3 s2 dày ‖ max-pool) ─ S8: RepCSP (4 khối, giãn 1,2,1,3)        stride 8
   ─ Down ─ S16: RepCSP (4 khối, giãn 1,2,1,3)                                   stride 16
   ─ Down ─ S32: RepCSP (2 khối) ─ SPPF (max-pool 5/9/13) ─ 1 lớp attention nhỏ  stride 32
Neck: PAN 2 chiều bằng RepCSP; chi tiết stride 4 vào P3 bằng space-to-depth
Head: GIỮ NGUYÊN của v8 (head dense theo hướng + ngữ cảnh trong tháp class + tầng suy luận quan hệ)
RepCSP: chia đôi kênh; nửa kia qua các RepConv (train: 3×3 + 1×1 + đồng nhất, mỗi nhánh có BN;
        suy luận: gộp thành MỘT conv 3×3); nối các đầu ra trung gian; 1×1
```

Pretrain (không cần nhãn), trước khi train detection:
- **Thầy 1:** lõi D-FINE-X COCO (HGNetv2-B5 + encoder lai, Apache-2.0), chạy ở cùng độ phân giải với học trò. Đầu ra P3 / P4 / P5 của thầy làm đích cho P3 / P4 / P5 của neck học trò.
- **Thầy 2:** DINOv2-S hoặc DINOv2-B (Apache-2.0). Đặc trưng patch của thầy làm đích cho P4 / P5 của học trò.
- **Loss:** 1 − cosine theo từng vị trí, qua một phép chiếu 1×1 cho mỗi cặp. Các phép chiếu bị bỏ sau pretrain.
- **Ảnh:** trộn ảnh tổng quát (một phần COCO), ảnh OBB (DOTA), ảnh bản vẽ (kho của user + ảnh train của Wall_Color, không cần nhãn).
  Ảnh cùng miền với ảnh sẽ dùng thật là bắt buộc (xem giả định G2).

## 3. Lập luận A: vì sao độ chính xác không kém v8

**Mệnh đề 1 (thừa kế biểu diễn qua chưng cất).**
- *Ký hiệu:*
  - T(x) là đặc trưng của thầy (lõi v3);
  - S(x) là đặc trưng của học trò, P là phép chiếu đã học;
  - pretrain đạt sai số ‖P(S(x)) − T(x)‖ ≤ ε trên phân phối ảnh D.
- *Khẳng định:* với mọi head h có hằng số Lipschitz L (head dense của v8), head h∘P đặt trên học trò cho đầu ra lệch so với h trên thầy không quá L·ε.
- *Hệ quả:* loss tốt nhất mà học trò + head có thể đạt ≤ loss tốt nhất của thầy + L·ε. Vì khi fine-tune, head được phép học bất kỳ hàm nào, trong đó có h∘P.

Mệnh đề này là bất đẳng thức tam giác; điều đáng giá nằm ở các **giả định**:

| Giả định | Nếu sai thì sao | Kiểm bằng gì |
|---|---|---|
| G1: kiến trúc học trò đủ sức xấp xỉ T (ε nhỏ) | ε lớn, cận vô nghĩa | Đo ε (cosine) trên ảnh giữ lại sau pretrain (phép thử T2) |
| G2: D giống phân phối ảnh khi dùng thật | ε trên ảnh thật lớn hơn ε khi pretrain | Đo cosine **riêng trên ảnh bản vẽ** chưa thấy khi pretrain |
| G3: fine-tune không phá phần đã thừa kế | Mất tiên nghiệm như v6 | Lr thấp cho lõi lúc đầu (như v3: backbone ×0.1); so đường cong với v8 |

**Vì sao có thể hơn v8, không chỉ bằng (lập luận bổ sung, yếu hơn Mệnh đề 1).**
- Hai thầy sai ở chỗ khác nhau (L5): DINOv2 giúp wall_300 và độ khít tường; HGNet / D-FINE giúp class hiếm và cửa (F33).
- Một học trò khớp cả hai (hai phép chiếu từ cùng một S) phải mang thông tin của cả hai. Head v8 khi fine-tune có thể dùng phần thông tin mà riêng v3 không có.
- Kết quả đã biết trong tài liệu chưng cất nhiều thầy là học trò có thể vượt từng thầy riêng lẻ khi các thầy bổ sung nhau. **Không đảm bảo**: đây là kỳ vọng có cơ sở, phép thử T3 quyết định.
- Lợi ích DINOv2 mà v4 phải trả 38 ms mới có, v9 lấy được **mà không chạy ViT khi suy luận**. Đây là điểm đột phá chính của v9: ngữ nghĩa cỡ foundation model ở tốc độ của mạng conv.

## 4. Lập luận B: vì sao nhanh hơn trên GPU

**Mô hình roofline.**
- Thời gian một lớp ≈ max(số phép tính / tốc độ tính, số byte đọc ghi / băng thông bộ nhớ).
- G4: khoảng 195 TFLOPS bf16 (đã đo, COLAB.md). Băng thông bộ nhớ khoảng 1.6–1.8 TB/s (thông số công bố, chưa tự đo).
- Ngưỡng cân bằng khoảng **110 FLOP / byte**. Lớp có cường độ tính (FLOP / byte) thấp hơn ngưỡng thì bị nghẽn bởi bộ nhớ.

| Loại lớp (bf16, 2 byte / phần tử) | FLOP mỗi điểm đầu ra | Byte mỗi điểm (đọc + ghi) | Cường độ | Dùng được bao nhiêu % GPU |
|---|---|---|---|---|
| Depthwise 3×3 | 18 | khoảng 4 | **khoảng 4.5** | ≤ 4% (nghẽn bộ nhớ) |
| Dải depthwise 1×11 | 22 | khoảng 4 | khoảng 5.5 | ≤ 5% |
| BN / activation / cộng riêng lẻ | 1–2 | khoảng 4 | < 1 | gần 0 (thuần chi phí bộ nhớ) |
| Conv 1×1, 256 kênh | 512 | khoảng 4 | khoảng 128 | quanh ngưỡng |
| **Conv 3×3 dày, 256 kênh** | 4608 | khoảng 4 | **khoảng 1150** | nghẽn tính toán, dùng tốt tensor core |

- Điều này giải thích L2: v6 có 504 GFLOPs nhưng phần lớn thời gian nằm ở các lớp depthwise / BN / cộng nghẽn bộ nhớ. Vì vậy nó chậm hơn v3 có 909 GFLOPs.
- **Tái tham số hoá** (RepVGG): tích chập là tuyến tính, nên 3×3 + 1×1 + đồng nhất (mỗi nhánh có BN) gộp được **chính xác** thành một conv 3×3 khi suy luận.
  Lúc train có nhiều nhánh (dễ tối ưu); lúc suy luận chỉ còn một lớp cường độ cao, không còn BN / cộng riêng lẻ.
- **Hệ quả kiểm được ngay:** độ trễ của v9 tỉ lệ gần với FLOPs của nó (cao hơn nhiều so với v6). Với ngân sách khoảng 500–600 GFLOPs (so với khoảng 970 của v8-x), độ trễ kỳ vọng khoảng 0.5–0.6× v8.
- **Phép thử T1 không cần train:** đo độ trễ bản đã gộp trên GPU với weights ngẫu nhiên. Tốn vài phút. Nếu không đạt ≤ 0.6× v8 thì sửa thiết kế trước khi tốn CU cho pretrain.
- **Vùng nhìn đủ rộng mà không cần depthwise:** cần khoảng 40–170 px ngữ cảnh (L4).
  - Các conv giãn (giãn 2, 3) trong RepCSP ở stride 8 và 16, cộng SPPF ở stride 32, cho vùng nhìn lý thuyết > 300 px ở P3. Chi phí như conv 3×3 thường.
  - Thêm một lớp attention nhỏ ở stride 32 (1600 token ở 1280 px), giữ "khoẻ" nhờ loss chưng cất ở P5 (mục 5).

## 5. Lập luận C: vì sao không lặp lại các lỗi đã đo

| Lỗi đã đo | Vì sao không xảy ra ở v9 |
|---|---|
| P5 chết, nổ số (F37) | (1) BN được train, không đóng băng. (2) Không có phép nhân `x·(1+m)` không chặn. (3) **P5 có loss chưng cất riêng** (đích là P5 của thầy). Một tầng chỉ "chết" khi không tín hiệu nào cần nó. Loss chưng cất ở P5 buộc P5 phải mang thông tin của P5 thầy. |
| Cổng LSK không chọn theo nội dung, tốn 60 GFLOPs cho conv 1×1 (F37) | Không dùng cổng chọn kernel. Ngữ cảnh bằng conv giãn + pooling, đều là phép đã biết là hiệu quả |
| Train từ đầu chững ở 0.55 (F40) | Không train từ đầu: pretrain truyền biểu diễn của lõi v3 (Mệnh đề 1) |
| Head nhẹ nên phân loại yếu (F37) | Giữ head của v8: có ngữ cảnh trong tháp class + tầng suy luận quan hệ (đã có lập luận ở OPENOBB8 mục 2) |
| Không tách được kiến trúc và khởi tạo (F40) | v9 chỉ khác v8 ở lõi (+ cách pretrain). Head, loss, công thức train giữ nguyên, nên so v9 với v8 là so đúng một yếu tố |

## 6. Kế hoạch kiểm chứng (dừng ngay khi bị bác bỏ)

| Bước | Phép thử | Đạt nếu | Chi phí ước tính | Nếu không đạt |
|---|---|---|---|---|
| T0 | Test CPU: gộp tái tham số hoá cho ra kết quả y hệt; mọi tầng có gradient; loss hữu hạn | Pass | 0 | Sửa code |
| **T1** | Độ trễ bản đã gộp, G4, 1280 px, batch 1, so với v8-x cùng điều kiện (weights ngẫu nhiên là đủ) | **≤ 0.6× v8** | khoảng 0.2 CU | Chỉnh độ rộng / số khối, đo lại; không pretrain khi chưa đạt |
| T2 | Pretrain 1 epoch để đo tốc độ, rồi pretrain đầy đủ; đo cosine học trò – thầy trên **ảnh bản vẽ giữ lại** | Cosine ≥ 0.8 ở P3 / P4 (ngưỡng đặt trước, chưa có số tham chiếu) | khoảng 15–25 CU (sẽ chốt từ tốc độ đo ở epoch 1) | Pretrain thêm / tăng độ rộng; nếu vẫn thấp thì G1 sai, dừng v9 |
| T3 | Train detection Wall_Color, cùng công thức với v8 | mAP50-95 ≥ v8 − 0.01, và wall_300 không thấp hơn v8 | khoảng 5–10 CU | Mệnh đề 1 không đủ trong thực tế; phân tích theo class trước khi làm gì thêm |
| T4 | Tổng quát: DOTA / FloorPlanCAD so YOLO26x | Ngang hoặc hơn YOLO26x, nhanh hơn | theo dataset | |

Chi phí ước tính dựa trên số đo F39 (17 ảnh/s cho v6-x ở 1280 trên G4) và roofline. Conv dày được kỳ vọng dùng GPU tốt hơn v6, nhưng con số thật chỉ có ở T1 / T2.

## 7. Vì sao không chọn hướng khác

| Hướng | Lý do loại |
|---|---|
| Dùng thẳng DINOv2 khi suy luận (như v4 / v5) | Chậm (38 ms, F33); patch 14 thô cho vật nhỏ |
| Pretrain kiểu YOLO (COCO 300 epoch có nhãn) | Quá đắt với ngân sách Colab (hàng trăm CU) |
| Pretrain tự giám sát thuần (masked image modeling) | Chậm hội tụ hơn chưng cất từ thầy đã mạnh; không truyền được kiến thức detection như thầy D-FINE |
| Giữ HGNetv2 và chỉ đổi phần khác | Đó là v8; không phải kiến trúc mới (user yêu cầu v9 phải mới) |
| Mạng nhiều depthwise / dải dài (như v6) | Nghẽn bộ nhớ trên GPU (mục 4, F39) |

## 8. Giới hạn của lập luận (nói trước)

- Mệnh đề 1 chỉ cho **cận trên của khoảng cách** với thầy, theo ε. Nó không hứa v9 tốt hơn. Phần "có thể hơn" dựa trên việc hai thầy bổ sung nhau (F33), là kỳ vọng chứ không phải chứng minh.
- Hằng số L của head không đo được dễ dàng. Vì vậy cận chỉ có ý nghĩa định tính: ε càng nhỏ thì càng gần thầy. Phép thử T2 đo ε, phép thử T3 đo kết quả cuối.
- Độ trễ kỳ vọng dựa trên roofline với băng thông lấy từ thông số công bố. T1 sẽ đo thật trước khi tốn CU.
- Thầy D-FINE là lõi của v3. Nếu chưng cất quá tốt và DINOv2 không thêm gì, v9 sẽ chỉ là **v8 nhanh hơn**. Đó vẫn là một kết quả có giá trị (sở hữu trọn, nhanh hơn), nhưng không phải đột phá về độ chính xác.
