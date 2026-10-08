# AGENTS.md: Nghiên cứu kiến trúc OBB mới cho bản vẽ CAD (vector PDF)

> File bàn giao cho mọi coding agent (Claude Code, Codex, Cursor, Antigravity...).
>
> **Tiếp tục nghiên cứu:** gọi skill **`vrdet-research`**. Claude Code: `/vrdet-research`. Codex: `$vrdet-research`. Agent khác: "đọc và làm theo `.agents/skills/vrdet-research/SKILL.md`". Skill tự đọc trạng thái, tự quyết thí nghiệm tiếp theo, tự chạy trên Colab và tự ghi lại kết quả.
> **Sổ cái nghiên cứu: [research/LEDGER.md](research/LEDGER.md).** Đọc trước khi chọn thí nghiệm. Mọi kết quả, phát hiện và lỗi
> đều phải ghi vào đó (luật của user, 2026-10-08).
>
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
- (bổ sung) **Mục tiêu sản phẩm, lời user:** "vừa thông minh như VLM vừa nhanh như YOLO".
  - VLM-det hiểu cả hình học lẫn ngữ nghĩa trên ảnh lớn 2000×3000, nhưng cực chậm và không sinh box song song.
  - YOLO nhanh nhưng gần như chỉ có hình học. Trong bản vẽ CAD, ngữ nghĩa nhiều khi quyết định object là gì.
  - Chấp nhận chậm hơn YOLO một chút, đổi lại hiểu ngữ nghĩa ảnh lớn **cùng** chi tiết nhỏ tốt hơn hẳn.
  - OBB phải chịu được object nhỏ và line mảnh.
  - Phải kiểm chứng trên **nhiều dataset public**: DOTA trước, rồi DIOR-R, HRSC2016, FloorPlanCAD.
  - Kiến trúc, code và weights lưu trong repo này, thuộc sở hữu của user, dùng thương mại được.
  - Hệ quả: H6 (ngữ cảnh toàn ảnh xuyên tile) và H3 (line mảnh) được nâng ưu tiên, ngay sau H4.
- (bổ sung 22:10) **Kiến trúc phải mới và thuộc sở hữu của user để thương mại hoá.**
  - Không dùng thư viện hạn chế thương mại như Ultralytics (AGPL) trong code sản phẩm.
  - Trường hợp tệ nhất vẫn chấp nhận: một kiến trúc mới độc quyền, độ chính xác và tốc độ na ná YOLO.
  - Đã chốt bằng `LICENSE` (proprietary), `THIRD_PARTY_NOTICES.md` (chỉ Apache-2.0: D-FINE/DEIM), `LICENSES/`, và `tests/test_license_guard.py` (cấm import ultralytics/mmcv/mmdet/mmrotate/ai4rs trong `vrdet/`).
- (bổ sung 21:50) **Đối thủ để so là cỡ lớn nhất: YOLO26x (và YOLO11x)**, không phải bản s.
  - VRDet được phép chậm hơn, nhưng phải chính xác và thông minh hơn.
  - **`dataset_obb_train_v3.zip` (floorplan trong repo) là data riêng để finetune sau. KHÔNG dùng để benchmark** cho tới khi kiến trúc đã thắng trên các dataset public (DOTA, FloorPlanCAD...).

- (bổ sung 23:20) User giao toàn quyền nghiên cứu, kể cả chọn G4 hay A100. Điều user quan tâm: một kiến trúc mới thông minh và nhanh, user sở hữu, finetune được trên data riêng tự gắn nhãn, đem đi thương mại. **Train phải tiết kiệm và tối ưu CU.**

- (bổ sung 23:35) **Tập trung kiến trúc trước; finetune bằng dataset riêng để sau.** Nghiên cứu tiếp cho tới khi Colab còn
  khoảng **120 CU**. Từ đó không mở session train mới, chỉ chạy nốt job đang dở. Sau đó **tạm chốt kiến trúc tốt nhất và dừng**
  để sáng 2026-10-08 user kiểm tra, đánh giá rồi mới làm tiếp. Đã cài vào watcher: `JOB_NEW_RESERVE=120`, job trong hàng đợi
  chỉ chạy nếu số dư trừ `est_cu` của job vẫn ≥ 120.

- (bổ sung 00:35, 2026-10-08) **Quy tắc CU qua đêm:** tới khoảng **170 CU** thì ngừng mở session mới. Job đang chạy dở được chạy nốt,
  miễn số dư cuối còn khoảng **120–150 CU**. User đi ngủ, giao nghiên cứu tự do, sáng sẽ kiểm tra.
  Watcher: `research/jobs/new_reserve.txt` = `170 130`. Job trong hàng đợi chỉ chạy khi số dư ≥ 170 **và**
  `số dư − est_cu − CU còn lại của các job đang chạy ≥ 130`. File này đọc mỗi lần kiểm tra, đổi không cần restart.

- (bổ sung 08:35, 2026-10-08) **Bỏ giới hạn 130–150/170 CU của đêm qua.** Nghiên cứu kiến trúc mới tiếp, chỉ giữ dự phòng gốc
  40 CU: `new_reserve.txt` = `50 40`. User đã đăng nhập lại Colab CLI.

- (bổ sung 09:30, 2026-10-08)
  - **Sổ cái:** mọi phát hiện phải ghi vào `research/LEDGER.md` để đọc lại và chọn hướng kế tiếp dựa trên kết quả cũ.
  - **Session:** chỉ đếm 3 session của agent, session người khác bật (`[?]`) không tính và không đụng tới.
    `cx.n_sessions()` mặc định đếm session của mình.

- (bổ sung 10:20, 2026-10-08) **Tận dụng ý tưởng từ MỌI kiến trúc tốt và mọi dataset mở, bất kể license** (kể cả NC/AGPL), vì đây là
  nghiên cứu học thuật. Code VRDet vẫn tự viết. Bản thương mại được user finetune trên data tự gắn nhãn.
  Đã giao 3 sub-agent: kiến trúc detection/OBB SOTA, ngữ nghĩa CAD + VLM grounding, dataset + pretrain đa dataset.

- (bổ sung 10:45, 2026-10-08) **Tốn CU quá: còn khoảng 80–100 CU thì DỪNG**, chốt gọn bản kiến trúc tốt nhất kèm cách train, để user tự train
  trên dataset riêng tự gán nhãn. Watcher: `new_reserve.txt` = `100 85`. Hàng đợi cắt còn e24, e12b, e17, e18, e13, e23.
  Bỏ e19 (LSK), e21 (chẩn đoán lớp), e22 (ortho). Không chạy run gộp cỡ X (khoảng 40 CU) vì thiếu ngân sách.
  Session A100 của người dùng chung cũng trừ vào số dư (khoảng 10 CU/giờ).

- (bổ sung 11:45, 2026-10-08) **Chỉ so với YOLO mạnh nhất (YOLO26x).** VRDet được phép nhỏ hơn; không bao giờ so với bản
  không phải SOTA như YOLO26s. Mỗi benchmark cần một run YOLO26x cùng điều kiện (FloorPlanCAD: `e8-fpc-yolo26x-24e`).

- (bổ sung 13:15, 2026-10-08) **QUAN TRỌNG: model chạy trên ẢNH bản vẽ CAD (raster), không có vector khi suy luận.**
  Mục tiêu: nhận diện object nhỏ, **đường ống mảnh**, và các khối trên ảnh.
  - DOTA chỉ là tham khảo cho lõi OBB; benchmark chính là FloorPlanCAD dạng ảnh, mốc YOLO26x 80.16/74.96.
  - Nhánh vector (c1/c3) chỉ là tuỳ chọn khi có PDF.
  - Việc cần làm cho kiến trúc: feature stride 4 (P2) cho nét mảnh / vật nhỏ, kernel dải cho ống dài thử trên CAD,
    render giữ nét. Raster-only tốt nhất hiện tại: e4 75.96/65.16; c4 (bản gộp không vector) đang chạy.

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
- [x] **E1a YOLO26s** (24 epoch, SS, train→val): **val mAP50 74.77**, mAP50:95 49.49. Sanity OK.
- [x] Smoke test VRDet trên T4 (dữ liệu giả lập): mAP50 1.00, pipeline đúng.
- [x] **E1b VRDet-S:** val mAP50 **69.89**, kém YOLO26s 4.9 điểm.
  - Thua ở class hiếm (HC, SBF, RA, BD, BC) do matching one-to-one cho quá ít mẫu dương ở lịch 24 epoch, và ở vật nhỏ dày (SV, SP) do trần 300 query.
  - Ngang YOLO ở class phổ biến. Latency 9.4 ms so với 8.9 ms.
- [x] **E2:**
  - H3 strip: −0.13, chưa kết luận.
  - H6 ngữ cảnh: −1.93, do lỗi BN của thumbnail (đã sửa).
  - H4 dense: e6 sub-mAP 63.3 so với 61.2, hứa hẹn.
  - H4c dense-query: e6 59.6.
  - H4 và H4c chạy trên A100 (chậm, 27 ảnh/s), xong khoảng 22:10.
- [x] **E3:** ngữ cảnh đã sửa BN: 69.71 (−0.18, class ngữ cảnh tăng nhưng HC −19). **RFS: 72.34 (+2.45), thành mặc định.** Khoảng cách tới YOLO26s còn −2.4.
- [x] **E4 FloorPlanCAD** (benchmark CAD thứ 2, val 810 bản vẽ, 30 class): YOLO26s 78.51, VRDet-S+RFS 75.96 (−2.55).
- [x] Kiểm kê phần mượn (23:30): VRDet-X 62.5M tham số = backbone 53% + encoder 33% của D-FINE (Apache) + decoder OBB 14% viết lại. Phần làm nên "kiến trúc mới thật" là nhánh vector + ngữ cảnh + head hybrid → đẩy H7 lên sớm.
- [x] **E5:** context + RFS 68.47 (−3.88) → **H6 thumbnail BÁC BỎ**. Head dense + RFS: decoder 71.84 (−0.51), riêng nhánh dense 49.8 (hỏng ở vật nhỏ dày đặc) → chưa dùng.
- [x] **E6 VRDet-X so với YOLO26x (DOTA val, 24 epoch):**
  - VRDet-X: **75.61 / 52.31**, latency 20.0 ms. YOLO26x: **78.42 / 53.10**, 12.8 ms.
  - Kém 2.81 mAP50 nhưng chỉ kém 0.79 mAP50:95. Đạt 64% FPS, thoả R3.
  - Dung tích lớn không tự đóng được khoảng cách.
- [x] **E7 nhánh vector (H7) trên FloorPlanCAD, cỡ S:** **76.80 / 67.67** (+0.84 / +2.51 so với raster). YOLO26s: 78.51 / 70.54.
- [x] **E9 trần query (H9):** suy luận với 600 query cho +0.7 (SV +6.3), latency không đổi → dùng 600 khi suy luận. Cờ mới `--eval-queries`.
- [ ] **Kiến trúc tạm chốt (2026-10-08):** VRDet = D-FINE-OBB (rotated FDR, Chamfer+KLD, MAL IoU xoay) + RFS + nhánh vector cho CAD + 600 query khi suy luận. Chưa vượt YOLO26 cùng cỡ. Báo cáo: [docs/REPORT_2026-10-08.md](docs/REPORT_2026-10-08.md).
- [ ] **Bước tiếp (đã soạn, chưa chạy):**
  - `e10-dota-mosaic-s`, `e10-fpc-vec-mosaic-s`: H8 mosaic, gỡ yếu tố gây nhiễu là YOLO có mosaic còn VRDet thì không.
  - Sau đó: train với 600 query; eval X với 600 query; sửa head dense.
  - **Chặn:** Colab CLI hết phiên đăng nhập OAuth (07:35). User phải đăng nhập lại; số dư khoảng 225 CU, không còn session nào.
- Bài học hạ tầng 2026-10-07:
  - Giữ tối đa 3 session (`JOB_MAX_SESSIONS=3`, đếm cả session lạ). Ba đợt mất VM (21:03, 22:18, 23:05) trùng lúc xin VM mới; user cho biết có lần là bị người khác tắt nhầm trên UI Colab. Session của agent tên dạng `e<N>-...-<attempt>`.
  - A100 chậm hơn G4 khoảng 2 lần với workload này (VM ít CPU). Head dense trên G4 + compile đạt 72 ảnh/s, gần bằng bản không dense (82).
  - `--compile` phải đặt sau khi tạo EMA (đã sửa).
  - Checkpoint lớn (YOLO26x 700 MB, VRDet-X ~1 GB) tải về theo chunk 16 MB, resume được qua nhiều lần poll (`cx.get`).
  - Máy local cạn commit memory (~2.5 GB) → torch CPU dễ crash; test nhỏ thì `OMP_NUM_THREADS=1`.

**GPU mặc định: G4** (RTX PRO 6000 Blackwell 94 GB, ~8.9 CU/h; 500 CU ≈ 55 giờ G4).

**Cần user:** điền tên chủ sở hữu pháp lý (cá nhân hoặc công ty) vào `LICENSE` (hiện để "the repository owner").

## 9. Nhật ký

- 2026-10-08 (đêm):
  - Mốc YOLO26x DOTA val (24 epoch) **78.42**. VRDet-X đạt **75.61** (−2.81; mAP50:95 −0.79; 64% FPS).
  - Context thumbnail bị bác bỏ (−3.88). Head dense chưa có lợi, bản thân nhánh dense hỏng.
  - Nhánh vector H7 trên FloorPlanCAD: +0.84 mAP50, +2.51 mAP50:95.
  - 600 query lúc suy luận cho +0.7 (SV +6.3).
  - Phát hiện yếu tố gây nhiễu mosaic (YOLO có, VRDet không), đã soạn E10.
  - Hạ tầng: giới hạn 3 session; download resume theo chunk; quy tắc CU 170/130; `tools/summarize.py`; `tools/import_yolo_obb.py` (finetune data riêng, để sau).
  - Tiêu khoảng 275 CU từ đầu, còn khoảng 225.

- 2026-10-07 (4):
  - Nghiên cứu web bằng 6 sub-agent. **Mốc mới: RiO-DETR** (ECCV 2026, Apache) đạt SS s 80.3 / x 81.8, nhỉnh hơn YOLO26.
  - Hai model mạnh ở chỗ khác nhau: DETR thắng class lớn cần ngữ cảnh, YOLO thắng vật nhỏ. Lấy max theo class được 82.3, nên chốt hướng **dense–sparse hybrid**.
  - Đổi license: DEIMv2 đã thành NC; code O2-RTDETR (ai4rs) chép từ RHINO (NC), nên chỉ lấy ý tưởng.
  - Viết evaluator, script data, VRDet v0, head dense. Launch baseline YOLO26s.

- 2026-10-07 (3): Xây khung tự động không cần người (cx.py, job.py, AUTONOMY.md, skill vrdet-research). Phát hiện: upload CLI rớt với file >~20–80 MB → chia chunk 16 MB + sha256; exec có thể treo khi 2 tiến trình poll cùng lúc → thêm lock + chỉ coi VM chết khi session biến mất hoặc 3 lần lỗi liên tiếp. G4 đo được 195 TFLOPS bf16. 2 lần smoke tiêu ~4 CU.

- 2026-10-07 (2): User chốt: Pro+ 500 CU, mục tiêu = kiến trúc OBB tổng quát benchmark trên DOTA như YOLO, accuracy trước. Viết RESEARCH_PLAN + COLAB. Colab CLI chạy được trên Windows sau 2 bản vá. Mốc phải vượt: YOLO26-OBB (x: 81.7 mAP50 DOTA test); DETR-OBB Apache tốt nhất đang biết: O2-DEIM-R50 80.15.

- 2026-10-07: Bắt đầu vòng nghiên cứu mới (VRDet). Kết luận vòng cũ thất bại do tự chế + from scratch. Đề xuất hướng hybrid vector+raster trên nền DETR-OBB Apache-2.0.
