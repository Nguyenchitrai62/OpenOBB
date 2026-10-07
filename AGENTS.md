# AGENTS.md: Nghiên cứu kiến trúc OBB mới cho bản vẽ CAD (vector PDF)

> File bàn giao cho mọi coding agent (Claude Code, Codex, Cursor, Antigravity...).
>
> **Tiếp tục nghiên cứu:** gọi skill **`vrdet-research`**. Claude Code: `/vrdet-research`. Codex: `$vrdet-research`. Agent khác: "đọc và làm theo `.agents/skills/vrdet-research/SKILL.md`". Skill tự đọc trạng thái, tự quyết thí nghiệm tiếp theo, tự chạy trên Colab và tự ghi lại kết quả.
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

## 4. Hướng kiến trúc (tóm tắt)

Tên làm việc: **VRDet**. Lõi = detector OBB raster tổng quát (benchmark DOTA). Nhánh vector = plug-in cho CAD. **Bản đầy đủ, cập nhật hơn: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)**; mục này chỉ là tóm tắt.

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
- **Colab CLI**: xem [docs/COLAB.md](docs/COLAB.md) (bắt buộc đọc). (`google-colab-cli` 0.7.4) đã cài bằng `uv tool install google-colab-cli`.
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

## 7. Quyết định của user (2026-10-07)

- Colab **Pro+**, ~500 CU. Không cần quá dè sẻn nhưng phải log CU mỗi thí nghiệm.
- **Mục tiêu chính là tạo kiến trúc OBB mới tổng quát**, theo đúng quy trình YOLO: thiết kế → pretrain trên dataset public → benchmark → user finetune trên data riêng tự gắn nhãn. Code vòng cũ (`cad_vlmdet`) chỉ là thử nghiệm, không phải nền.
- Dataset public chỉ dùng cho nghiên cứu/benchmark, nên license NC/học thuật (DOTA, FloorPlanCAD) dùng được. Backbone pretrained ImageNet cũng được.
- Ưu tiên: **độ chính xác trước**, tốc độ sau (kỳ vọng gần YOLO, chậm hơn chút được).
- → Benchmark chính: **DOTA-v1.0** (giống YOLO-OBB). Chi tiết: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md).
- (bổ sung, cùng ngày) **Agent có toàn quyền chọn kiến trúc. Chưa đạt mục tiêu "kiến trúc mới hơn hẳn các kiến trúc hiện có" thì không dừng.**
  - Dataset học thuật được dùng thoải mái để pretrain và benchmark; user có dataset chuyên biệt để finetune khi dùng thật.
  - Thử một hoặc vài dataset trước, khi kiến trúc ổn mới mở rộng.
  - Tối ưu cả thời gian lẫn CU. Dùng sub-agent để nghiên cứu song song, lấy cảm hứng từ sản phẩm có sẵn.
  - Được lấy **ý tưởng** từ cả kiến trúc public nhưng cấm thương mại, miễn là tự viết lại toàn bộ (không chép code, không dùng weights NC).

Câu hỏi còn mở:
- Nộp kết quả DOTA test cần tài khoản trên server đánh giá DOTA (user tạo khi đến E4). Trước đó ablation chỉ dùng val.

## 8. Trạng thái hiện tại

- [x] Khảo sát vòng cũ + AI_Takeoff + 2 repo tham khảo
- [x] Colab CLI trên Windows (2 bản vá, [docs/COLAB.md](docs/COLAB.md)), user đã đăng nhập (`nam@pose3d.ai`)
- [x] Khung tự động [tools/cx.py](tools/cx.py) + [tools/job.py](tools/job.py) ([docs/AUTONOMY.md](docs/AUTONOMY.md)). Đã test đầy đủ trên Colab thật: T4 (`e0-smoke`) và G4 (`e0-smoke-g4`), gồm launch → sync → kill VM → tự tạo VM mới + resume (giữ lịch sử metrics) → done → tự stop VM.
- [x] Skill `vrdet-research` (`.agents/skills/`, adapter `.claude/skills/`)
- [x] Bỏ Google Drive (mount cần người bấm Allow mỗi VM). Thay bằng: data tải trực tiếp trên VM, checkpoint đồng bộ về `runs/` local.
- [x] **E0 xong:**
  - `vrdet/eval/dota.py`: evaluator theo đúng protocol devkit (VOC07, bỏ difficult, IoU polygon, ghép patch bằng NMS 0.1), có test.
  - `colab/data/get_dota.py` + `split_dota.py`: tải DOTA-v1.0 gốc từ HF `Last-Bullet/DOTAv1.0` (pinned), cắt 1024/200 ra khoảng 15.8k patch train, mất khoảng 6 phút trên VM.
  - `colab/baselines/yolo_obb.py`: baseline Ultralytics, chỉ chạy trên Colab.
- [x] **Code VRDet v0** (`vrdet/`):
  - D-FINE-S (Apache) chuyển sang OBB với rotated FDR (4 cạnh + phân phối góc), cost Chamfer + KLD, MAL dùng IoU xoay chính xác, init từ COCO.
  - Head dense xoay + rotated TAL (H4) bật bằng `--dense`; eval ghép đầu ra theo nhiều kiểu.
  - Trainer `vrdet/train.py`: EMA, LR flat-cosine, resume.
  - Test: `pytest -q tests/` (CPU).
  - **Lưu ý: torch 2.12 CPU trên máy này crash (heap) khi chạy nhiều luồng.** Chạy CPU thì dùng `--threads 1`, hoặc test trên Colab.
- [ ] **E1 (đang chạy):**
  - `e1-yolo26s-dota-24e` (G4): baseline cùng điều kiện, khoảng 40 phút.
  - `e1-vrdet-smoke-t4` (T4): test chức năng VRDet trên dữ liệu giả lập.
  - Sau đó: `e1-vrdet-s-dota-24e` (G4).
- [ ] E2: H4 dense hybrid (`--dense`), rồi H1 ProbIoU, H2, H3 (xem `research/HYPOTHESES.md`).

**GPU mặc định: G4** (RTX PRO 6000 Blackwell 94 GB, ~8.9 CU/h; 500 CU ≈ 55 giờ G4).

**Cần user:** (trống)

## 9. Nhật ký

- 2026-10-07 (4):
  - Nghiên cứu web bằng 6 sub-agent. **Mốc mới: RiO-DETR** (ECCV 2026, Apache) đạt SS s 80.3 / x 81.8, nhỉnh hơn YOLO26.
  - Hai model mạnh ở chỗ khác nhau: DETR thắng class lớn cần ngữ cảnh, YOLO thắng vật nhỏ. Lấy max theo class được 82.3, nên chốt hướng **dense–sparse hybrid**.
  - Đổi license: DEIMv2 đã thành NC; code O2-RTDETR (ai4rs) chép từ RHINO (NC), nên chỉ lấy ý tưởng.
  - Viết evaluator, script data, VRDet v0, head dense. Launch baseline YOLO26s.

- 2026-10-07 (3): Xây khung tự động không cần người (cx.py, job.py, AUTONOMY.md, skill vrdet-research). Phát hiện: upload CLI rớt với file >~20–80 MB → chia chunk 16 MB + sha256; exec có thể treo khi 2 tiến trình poll cùng lúc → thêm lock + chỉ coi VM chết khi session biến mất hoặc 3 lần lỗi liên tiếp. G4 đo được 195 TFLOPS bf16. 2 lần smoke tiêu ~4 CU.

- 2026-10-07 (2): User chốt: Pro+ 500 CU, mục tiêu = kiến trúc OBB tổng quát benchmark trên DOTA như YOLO, accuracy trước. Viết RESEARCH_PLAN + COLAB. Colab CLI chạy được trên Windows sau 2 bản vá. Mốc phải vượt: YOLO26-OBB (x: 81.7 mAP50 DOTA test); DETR-OBB Apache tốt nhất đang biết: O2-DEIM-R50 80.15.

- 2026-10-07: Bắt đầu vòng nghiên cứu mới (VRDet). Kết luận vòng cũ thất bại do tự chế + from scratch. Đề xuất hướng hybrid vector+raster trên nền DETR-OBB Apache-2.0.
