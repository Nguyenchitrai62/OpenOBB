# AGENTS.md: Nghiên cứu kiến trúc OBB mới cho bản vẽ CAD (vector PDF)

> File bàn giao cho mọi coding agent (Claude Code, Codex, Cursor, Antigravity...).
> **Đọc hết file này trước khi làm gì.** Cập nhật mục "Trạng thái hiện tại" và "Nhật ký" mỗi khi xong một bước, để agent sau làm tiếp được.

## 1. Mục tiêu

Thiết kế và huấn luyện một **kiến trúc detector OBB mới** để nhận diện object trong bản vẽ PDF vector export từ CAD (dùng cho repo `F:\Source_code\AI_Takeoff`). Kiến trúc mới phải:

| Mã | Yêu cầu |
|---|---|
| R1 | **Độ chính xác cao hơn YOLO-OBB** khi so cùng điều kiện (cùng data, cùng imgsz, cùng epoch budget) |
| R2 | **Hiểu ngữ nghĩa**: bản vẽ CAD có ngữ nghĩa cao (layer, text, ký hiệu, topology), còn YOLO chỉ thuần hình học |
| R3 | **Nhanh gần bằng YOLO**: chấp nhận chậm hơn một chút, mục tiêu ≥60% FPS của YOLO cùng cỡ |
| R4 | **Thương mại được**: code + weights không dính AGPL (Ultralytics). Chỉ dùng thành phần MIT/Apache-2.0/BSD |
| R5 | Pretrain + benchmark được trên dataset public, giống cách YOLO dùng COCO/DOTA |

Ultralytics chỉ được dùng làm **baseline để so sánh**, không bao giờ ship.

## 2. Bối cảnh sản phẩm (AI_Takeoff), đã khảo sát 2026-10-07

- AI_Takeoff **không chạy YOLO trong repo**: mọi detection gọi HTTP sang model server ngoài (`ai_gate` → `/yolo/*`, `/model-tools/predict`), SAHI tile 1024², overlap 0.3–0.5. Version/size của YOLO nằm trên server, repo không thấy.
- Lĩnh vực: **MEP**: HVAC (ổn định), FIRE/sprinkler (ổn định), plumbing, ELCV điện nhẹ (đang phát triển).
  - FIRE: OBB classes `junction`, `connect` (line). HVAC: YOLO HBB symbols (Fan, Soft Pipe), ducts/fittings suy ra từ hình học vector.
  - ELCV: camera, wifi, modem, hộp nối, cable tray... Plumbing: layout (axis, legend, table, note, title_block) + thiết bị vệ sinh.
- PDF → ảnh: PyMuPDF, mặc định scale 3× (216 DPI).
- **Repo đã trích vector rất kỹ**: paths + tên layer/OCG (`BE/app/drawing_cache.py`, `BE/app/layers.py`), text (`page.get_text`). Pipeline HVAC chạy trực tiếp trên vector, còn model chỉ xác nhận.
- Bằng chứng quan trọng (`AI_Takeoff/experiments/vlm_filter/README.md`): **FP và TP nhìn giống hệt nhau trên ảnh**. Phân biệt được là nhờ hình học/topology/text (ví dụ "không có DN" → 114/136 là FP). Đây là lý do chính để kiến trúc mới phải ăn được **vector + text**, không chỉ pixel.
- Metric sản phẩm hiện tại là F2 pipeline (HVAC ~0.90, FIRE ~0.91), không có mAP.
- GT được review trên UI rồi export YOLO (HVAC: HBB, FIRE: OBB) render 3× kèm manifest (`BE/routers/training_exports.py`). Nghĩa là **có cặp PDF gốc + nhãn**, nên trích được vector cho training.

## 3. Di sản: vòng nghiên cứu trước (CAD-VLMDet, 2026-09-22 → 09-24, Kaggle T4). THẤT BẠI, phải học bài

Thư mục: `cad_vlmdet/` (code), `kaggle_pkg/` (kernel + log + weights), `research/` (HYPOTHESES.md, decisions.log).

- Data: `dataset_obb_train_v3.zip` (1.6 GB, **floorplan**: 16.158 train / 852 valid, 7 classes `wall, window, door, slide_door, double_door, opening, junction`, YOLO-OBB 8-điểm normalized, ~150 box/ảnh, max 396). Subset 800/200 ở `cad_vlmdet/subset`.
- Kiến trúc tự chế: ConvNeXt + strip-pool GCAM + transformer P5 + head OBB dense + topology loss, train **from scratch**.
- Kết quả (800 ảnh, 768px, 20ep): **mean AP50 = 0.144** (wall 0.12, window 0.10, door/slide/double = **0**, junction 0.79). YOLO11x-OBB cùng điều kiện: **mAP50-95 ≈ 0.585**. Thua rất xa.
- Nguyên nhân rút ra:
  1. Không có pretrained weights + data nhỏ + 20 epoch: kiến trúc lạ không hội tụ kịp.
  2. Tự viết assigner/loss/decode chưa kiểm chứng: door collapse (model học "door luôn vắng mặt"), focal loss sai, level-assign sai.
  3. Evaluator là "HBB-proxy" tự viết, không phải rotated-mAP chuẩn DOTA.
  4. Thay nhiều thứ cùng lúc, không reproduce baseline trước.
- Điều đã xác nhận có ích: suy luận toàn cục (transformer) giúp window +5đ, junction +3đ (ablation H2), còn nhánh stride-4 giúp junction.

**Luật rút ra (BẮT BUỘC):**
- Xây trên codebase detector **đã chứng minh, license Apache/MIT**. Bước đầu phải **reproduce được số baseline công bố** trước khi thêm cái mới.
- Mỗi thí nghiệm chỉ đổi **một** yếu tố, luôn có baseline chạy cùng điều kiện.
- Evaluator: rotated mAP50 / mAP50-95 chuẩn (kiểu DOTA devkit / mmrotate), cộng thêm F2 theo class cho bài toán sản phẩm.
- Không claim vượt trội khi chưa có số.

## 4. Hướng kiến trúc đề xuất (bản nháp, chờ user chốt các câu hỏi ở mục 7)

Tên làm việc: **VRDet** (Vector-Raster Detector).

```
PDF page
 ├─ Raster branch: render 2–3× → backbone CNN/ViT nhẹ, pretrained (Apache) → FPN P2–P5
 ├─ Vector branch: PyMuPDF get_drawings/get_text → primitive tokens
 │     (line/arc/bezier: toạ độ, độ dày nét, màu, dash, layer-name embedding, text-span embedding)
 │     → point/line transformer nhẹ (ý tưởng từ SymPoint-V2 / VecFormer / CADSpotting)
 ├─ Fusion: cross-attention vector-token ↔ raster-feature (hoặc bản rẻ: render thêm
 │     kênh ngữ nghĩa: arc-map, linewidth-map, layer-hash-map, text-mask)
 └─ Head: real-time DETR kiểu D-FINE/DEIM (Apache-2.0) mở rộng sang OBB
       (box = cx,cy,w,h,θ, refine dạng phân phối; dense O2O + one-to-many aux
        để chịu được 150–400 object/ảnh; NMS-free)
```

Lý do:
- DETR self-attention giữa các query mô hình hoá được **quan hệ** (cửa nằm trên tường, sprinkler nằm trên line, tee là giao 3 line), thứ YOLO local-CNN không có.
- Vector branch cho mô hình thấy đúng thứ mà pipeline AI_Takeoff hiện xử lý bằng luật tay (layer, DN text, hình học).
- Fusion "bản rẻ" (kênh ngữ nghĩa raster) gần như không tốn tốc độ, nên làm trước để đo lợi ích, rồi mới làm cross-attention.

Lộ trình (mỗi bước có tiêu chí pass/fail, ghi vào `research/HYPOTHESES.md`):
1. **E0 hạ tầng**: Colab CLI + Drive checkpoint + evaluator rotated-mAP chuẩn + baseline YOLO11-OBB (chỉ để đo).
2. **E1 reproduce**: OBB-DETR head trên DOTA-v1 (subset) đạt gần số công bố của codebase gốc.
3. **E2 raster-only** trên data CAD: phải ≥ YOLO-OBB cùng điều kiện.
4. **E3 +kênh ngữ nghĩa vector** (bản rẻ): đo Δ theo class.
5. **E4 +vector tokens cross-attention + text**: đo Δ, đo FPS.
6. **E5 scale + distill + export ONNX/TensorRT**, đo FPS so YOLO.

Dataset public tham khảo (**chú ý license**):
| Dataset | Dùng cho | License |
|---|---|---|
| DOTA-v1.0/1.5 | benchmark OBB chuẩn (YOLO-OBB dùng) | chỉ học thuật |
| FloorPlanCAD (vector SVG, 30+ class) | benchmark symbol spotting có vector | CC BY-NC 4.0 |
| CubiCasa5K | floorplan raster | CC BY-NC 4.0 |
| ArchCAD-400K | CAD vector lớn | cần kiểm tra |
| COCO / Objects365 | pretrain chung | CC BY 4.0 (annotation) |
| Synthetic CAD tự sinh | pretrain sạch license | của mình |

Weights thương mại cuối cùng nên train trên data sạch (data nội bộ + synthetic). Data NC chỉ dùng để benchmark/nghiên cứu, trừ khi user chấp nhận rủi ro.

## 5. Môi trường & công cụ

- Máy local: Windows 11, **không có GPU NVIDIA** (Intel UHD 730), RAM 32 GB, ổ F còn ~50 GB. Train **chỉ trên Colab** (hoặc Kaggle nếu user đồng ý).
- Python: `py -3.12` (có pip), `uv` ở `C:\Users\HP\.local\bin\uv.exe`. Python 3.11 hệ thống **không có pip**. Torch chưa cài local.
- WSL Ubuntu **hỏng** (thiếu ext4.vhdx). Docker Desktop không chạy.
- **Colab CLI** (`google-colab-cli` 0.7.4) đã cài bằng `uv tool install google-colab-cli`.
  - Bản gốc không chạy trên Windows (`import termios`). **Đã vá**: `C:\Users\HP\AppData\Roaming\uv\tools\google-colab-cli\Lib\site-packages\colab_cli\console.py`, bọc `termios`/`tty` trong try/except (bản gốc lưu ở `console.py.orig`). Mọi lệnh chạy được trừ `colab console`. Nếu `uv tool upgrade` thì phải vá lại.
  - Trên Windows nên đặt `PYTHONIOENCODING=utf-8` khi gọi từ Git Bash.
  - Lệnh chính: `colab new -s <name> --gpu T4|L4|A100|H100`, `colab exec -s <name> -f script.py`, `colab upload/download -s <name> LOCAL REMOTE`, `colab drivemount -s <name>`, `colab install -s <name> -r requirements.txt`, `colab status`, `colab usage`, `colab stop -s <name>`, `colab run --gpu A100 script.py` (VM tạm).
  - Auth: mặc định oauth2 (user tự đăng nhập trên trình duyệt). **Agent không tự nhập mật khẩu.**
  - Session Colab có thể chết bất cứ lúc nào, nên **checkpoint mỗi epoch vào Google Drive** và train phải resume được.
- OpenResearch (alphaXiv): harness nghiên cứu local-first (`orx up`, dashboard 127.0.0.1:4791, lưu experiment vào SQL). Bản Windows là beta exe, **user tự cài nếu muốn**. Repo này theo cùng phương pháp: giả thuyết → thí nghiệm rẻ → ghi log.
- Secrets: `.env` (kaggle key) và `AI_Takeoff/.env`. **Không bao giờ in, commit, hay gửi đi.**

## 6. Cấu trúc thư mục

```
AGENTS.md                 ← file này (nguồn sự thật cho handoff)
research/                 ← HYPOTHESES.md, decisions.log, DIRECTIONS.md (vòng cũ + mới)
cad_vlmdet/               ← code vòng cũ (tham khảo, KHÔNG phát triển tiếp)
kaggle_pkg/               ← kernel/log/weights vòng cũ
dataset_obb_train_v3.zip  ← data floorplan (gitignored)
vrdet/                    ← (sẽ tạo) code kiến trúc mới
colab/                    ← (sẽ tạo) script chạy trên Colab
```

## 7. Câu hỏi mở cho user (chặn một phần công việc)

1. Gói Colab: Free / Pro / Pro+? Còn bao nhiêu compute units? Có cho dùng thêm Kaggle (30h T4/tuần) không?
2. **Domain mục tiêu**: data trong thư mục này là floorplan (wall/door/window), còn AI_Takeoff là MEP (FIRE junction/connect, HVAC symbols, ELCV symbols). Ưu tiên domain nào?
3. Có cấp được **PDF gốc + nhãn GT** export từ AI_Takeoff không (để có vector)? Nguồn gốc/license của `dataset_obb_train_v3`?
4. Mức chặt license: được dùng backbone pretrained ImageNet (timm, Apache) không? Data NC (FloorPlanCAD, DOTA) chỉ để benchmark có ổn không?
5. Mục tiêu tốc độ: inference chạy GPU gì (hay CPU)? Bao nhiêu giây/trang?

## 8. Trạng thái hiện tại

- [x] Khảo sát vòng cũ + AI_Takeoff + 2 repo tham khảo
- [x] Cài + vá Colab CLI cho Windows
- [x] Viết AGENTS.md, init git
- [ ] User đăng nhập Colab (`colab usage` để kích hoạt OAuth)
- [ ] User trả lời mục 7
- [ ] E0 hạ tầng

## 9. Nhật ký

- 2026-10-07: Bắt đầu vòng nghiên cứu mới (VRDet). Kết luận vòng cũ thất bại do tự chế + from scratch. Đề xuất hướng hybrid vector+raster trên nền DETR-OBB Apache-2.0.
