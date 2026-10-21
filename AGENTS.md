# AGENTS.md: Nghiên cứu kiến trúc OBB mới cho bản vẽ CAD (vector PDF)

> File bàn giao cho mọi coding agent (Claude Code, Codex, Cursor, Antigravity...).
>
> **Tiếp tục nghiên cứu:** gọi skill **`vrdet-research`**. Claude Code: `/vrdet-research`. Codex: `$vrdet-research`. Agent khác: "đọc và làm theo `.agents/skills/vrdet-research/SKILL.md`". Skill tự đọc trạng thái, tự quyết thí nghiệm tiếp theo, tự chạy trên Colab và tự ghi lại kết quả.
> **Quy trình nghiên cứu chuẩn (bắt buộc cho mọi agent, mọi session): mục 8b.**
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
openobb/                    ← (sẽ tạo) code kiến trúc mới
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
  - Đã chốt bằng `LICENSE` (proprietary), `THIRD_PARTY_NOTICES.md` (chỉ Apache-2.0: D-FINE/DEIM), `LICENSES/`, và `tests/test_license_guard.py` (cấm import ultralytics/mmcv/mmdet/mmrotate/ai4rs trong `openobb/`).
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

- (bổ sung 13:30, 2026-10-08) **Bỏ giới hạn 80–100 CU.** Chỉ giữ dự phòng gốc 40 CU (`new_reserve.txt` = `50 40`).
  Tiếp tục nghiên cứu kiến trúc cho ảnh CAD (raster).

- (bổ sung 13:35, 2026-10-08) **BỎ nhánh vector.** Chỉ làm OBB detection trên ảnh (raster). Code vector (`--vectors`,
  layer pooling, box neo theo nét c3) giữ trong repo nhưng ngoài phạm vi; c3 đã dừng giữa chừng để tiết kiệm CU.

- (bổ sung 16:30, 2026-10-08) **Bộ khung train/infer phải tiện như YOLO; lấy triệt để ý tưởng của YOLO** (ý tưởng không bị
  bản quyền; code tự viết, không chép Ultralytics). Kiến trúc model là của VRDet.
  - Mặc định: mỗi ảnh resize về cạnh dài `imgsz`. Mosaic 1.0, scale 0.5, translate 0.1, close_mosaic 10 (mosaic kiểu YOLO
    trên CAD chưa đo).
  - Warmup 3 epoch, patience 100, tự giảm batch khi OOM, val mỗi epoch.
  - Báo cáo: `results.csv/png`, `labels.jpg`, `val_pred.jpg`. Kiểm tra nhãn lúc nạp.
  - Cắt tile giờ chỉ là tuỳ chọn `tile=True`.
  - Data Wall_Color: user tự train trên Colab bằng `colab/OpenOBB_wall_color.ipynb`.
  - (bổ sung chiều 10-08, theo user) **Thao tác y như YOLO:**
    - Mỗi lần train là một thư mục mới `runs/obb/train`, `train2`...; weights nằm trong `weights/best.pt` và `weights/last.pt`.
    - **Không tự resume**: chỉ resume khi có `resume=True`.
    - `lr0` truyền vào là dùng đúng giá trị đó. Lưu ý: YOLO với `optimizer=auto` thì bỏ qua lr0; ở đây user chọn khác.
    - Predict lưu vào `runs/obb/predict`, `predict2`...

- (bổ sung 2026-10-16) **Khung training = khung YOLO, không có khung riêng.** Ý tưởng gốc: khung train / infer giống hệt YOLO, chỉ kiến trúc
  model là của OpenOBB. Từ 0.23.0 khung mặc định duy nhất là của Ultralytics 8.4 (AdamW lr0 = 0.002·5/(4+nc), LR tuyến tính, wd 5e-4,
  EMA 0.9999/2000, nbs 64, HSV 0.7/0.4, không lật dọc / xoay 90°, không RFS); bỏ `trainer=` và các tự chỉnh riêng cũ. `lr0` truyền tay vẫn
  dùng đúng giá trị. **`unlock_all` giữ lại như tính năng riêng của kiến trúc** (mở khoá thân khi có data lớn). Căn cứ: F103–F104.
  (bổ sung cùng ngày, 0.24.0) **Một điểm khác YOLO có chủ đích: LR giảm theo cosine là mặc định** (`cos_lr=True`; `cos_lr=False` = tuyến tính
  như YOLO). Lý do của user: đầu run giữ LR cao để dò vùng lõm, nửa sau hạ nhanh để vào tâm. Đánh giá theo đồ thị LR: F108.

Câu hỏi còn mở:
- Nộp kết quả DOTA test cần tài khoản trên server đánh giá DOTA (user tạo khi đến E4). Trước đó ablation chỉ dùng val.

## 8. Trạng thái hiện tại

- [x] Khảo sát vòng cũ + AI_Takeoff + 2 repo tham khảo
- [x] Colab CLI trên Windows (2 bản vá, [docs/COLAB.md](docs/COLAB.md)), user đã đăng nhập (`nam@pose3d.ai`)
- [x] Khung tự động [tools/cx.py](tools/cx.py) + [tools/job.py](tools/job.py) ([docs/AUTONOMY.md](docs/AUTONOMY.md)). Đã test đầy đủ trên Colab thật: T4 (`e0-smoke`) và G4 (`e0-smoke-g4`), gồm launch → sync → kill VM → tự tạo VM mới + resume (giữ lịch sử metrics) → done → tự stop VM.
- [x] Skill `vrdet-research` (`.agents/skills/`, adapter `.claude/skills/`)
- [x] Bỏ Google Drive (mount cần người bấm Allow mỗi VM). Thay bằng: data tải trực tiếp trên VM, checkpoint đồng bộ về `runs/` local.
- [x] **E0 xong:**
  - `openobb/eval/dota.py`: evaluator theo đúng protocol devkit (VOC07, bỏ difficult, IoU polygon, ghép patch bằng NMS 0.1), có test.
  - `colab/data/get_dota.py` + `split_dota.py`: tải DOTA-v1.0 gốc từ HF `Last-Bullet/DOTAv1.0` (pinned), cắt 1024/200 ra khoảng 15.8k patch train, mất khoảng 6 phút trên VM.
  - `colab/baselines/yolo_obb.py`: baseline Ultralytics, chỉ chạy trên Colab.
- [x] **Code VRDet v0** (`openobb/`):
  - D-FINE-S (Apache) chuyển sang OBB với rotated FDR (4 cạnh + phân phối góc), cost Chamfer + KLD, MAL dùng IoU xoay chính xác, init từ COCO.
  - Head dense xoay + rotated TAL (H4) bật bằng `--dense`; eval ghép đầu ra theo nhiều kiểu.
  - Trainer `openobb/train.py`: EMA, LR flat-cosine, resume.
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
- [x] **KIẾN TRÚC CHỐT (2026-10-08 14:50), raster-only: VRDet-S + LSK** (`c6-fpc-raster-lsk-s`). Thẻ: [docs/OPENOBB1.md](docs/OPENOBB1.md).
  - Thành phần: D-FINE-OBB + adapter LSK + RFS + nhóm query một-nhiều 900 + AQD + loss góc + IoU-cost + 900 query khi suy luận.
  - FloorPlanCAD val: **78.16 / 68.81** (12.5M tham số, 11.4 ms) so với YOLO26x 80.16 / 74.96 (57.6M, 11.6 ms). Recall ngang; thua chủ yếu độ khít box object mảnh (F18).
  - Thư viện pip + CLI: `openobb train|val|predict|prepare` (`openobb/cli.py`, API `vrdet.Detector`), mặc định là công thức này.
    Notebook Colab của user: `colab/OpenOBB_train.ipynb`, cài từ GitHub `Nguyenchitrai62/OpenOBB` (public). Hướng dẫn: [docs/FINETUNE.md](docs/FINETUNE.md).
  - Đã dừng: watcher tắt, không còn session Colab, số dư khoảng 71.7 CU. Hướng kế tiếp (chờ user): xem cuối F18 trong sổ cái.
- [x] (cũ) Kiến trúc tạm chốt sáng 10-08: D-FINE-OBB + RFS + nhánh vector + 600 query. Báo cáo: [docs/REPORT_2026-10-08.md](docs/REPORT_2026-10-08.md).
- [x] **VRDet2 (2026-10-09), kiến trúc mới:**
  - Dense segment-aware, P2–P5, không NMS, 55.6M, tự viết 100%, train từ đầu. Chi tiết: [docs/OPENOBB2.md](docs/OPENOBB2.md).
  - Dùng: `model=vrdet2x`, CLI giữ nguyên. `model=x` = `vrdet1x` là kiến trúc cũ.
  - User tự train và gửi log. Agent chỉ thiết kế kiến trúc.
  - Mỗi version kiến trúc mới phải nêu chi tiết từng phần và tỉ lệ tận dụng cái có sẵn (luật user, 10-09).
- [x] **VRDet3 (2026-10-09):**
  - Encoder pretrained COCO (như v1) + LSK + head dense oriented dị hướng (DFL dọc trục / độ dày), NMS. 64.6M tham số.
  - Dùng: `model=vrdet3x`. Chi tiết: [docs/OPENOBB3.md](docs/OPENOBB3.md).
  - Lý do: v2 train từ đầu chỉ đạt 0.27 / 0.10 → pretrained quyết định trên data nhỏ; `end_loss` của v2 bị kẹt (LEDGER F25–F26).
- [x] **Thước đo kiểu Ultralytics (2026-10-09, F29, F31):** mặc định cho val mỗi epoch và chọn best.pt (`metric=yolo`); thước DOTA in kèm.
  - Cùng thước: **VRDet3-x 0.720 / 0.474 so với YOLO11x 0.74 / 0.50.**
- [x] **VRDet4-x (2026-10-09, F30):**
  - DINOv2 ViT-B + adapter + LSK + encoder/decoder DETR COCO + nhánh dense oriented (head v3), đầu ra union. 134.8M tham số.
  - Dùng: `model=vrdet4x`. Chi tiết: [docs/OPENOBB4.md](docs/OPENOBB4.md). User chỉ train bản x.
- [x] **VRDet5-x (2026-10-09, F34):**
  - Head dense v3 + DINOv2 + phân loại theo hình học + chấm lại theo quan hệ. 127.9M tham số.
  - Dùng: `model=vrdet5x`. Chi tiết: [docs/OPENOBB5.md](docs/OPENOBB5.md).
  - Val in thêm confusion matrix (lỗi nhầm class thường gặp).
- Kết quả cùng thước YOLO (Wall_Color): YOLO26x 0.746 / 0.499; v3 0.712 / 0.467; v4 0.663 / 0.457 (1 run). Xem LEDGER F32–F33.
- **Mốc chính thức 2026-10-14 (F59, log user): YOLO26x 0.746 / 0.499 (150 ep, A100); OpenOBB12-x head một-nhiều + NMS 0.7: 0.743 / 0.472, 15.3 ms / ảnh (G4, batch 16).**
  Khoảng cách nằm ở mAP50-95 của wall / wall_300 / junction / note (độ khít box) và mAP50 của wall_300 (phân loại); class hiếm v12 thắng.
- [x] **OpenOBB14 (2026-10-14, F58):** thân HGNetv2 pretrained giữ nguyên + neck / head / bộ tinh chỉnh box tự thiết kế.
  Chi tiết và kỳ vọng ghi trước: [docs/OPENOBB14.md](docs/OPENOBB14.md).
  **Kết quả (F60):** 0.713 / 0.469, 10.9 ms (v12: 0.743 / 0.472, 15.3 ms). Tốc độ đạt, wall +.045 mAP50-95, nhưng note / slide_door /
  wall_300 tụt → mất tiên nghiệm encoder. Đang chờ phép thử `neck=dfine` (0.13.1) để tách nguyên nhân.
- [x] **OpenOBB15 (2026-10-14, F61, 0.14.0):** v14 + luồng độ phân giải gốc + lưới P2 + bộ tinh chỉnh đọc luồng gốc, nhắm mAP50-95 của
  junction / tường / wall_300 trên Wall_Color. [docs/OPENOBB15.md](docs/OPENOBB15.md). Vòng hai hạt giống: `openobb15x neck=dfine` so với
  `openobb14x neck=dfine`. Chờ log user.
  **Kết quả bản `neck=own` (F63):** 0.653 / 0.416, 14.5 ms (batch 8). wall / junction / wall_300 đứng yên so với v14 own, note sụp
  (−.245 mAP50-95). Cặp đăng ký `neck=dfine` vẫn chưa chạy.
- [x] **OpenOBB16 (2026-10-14, F62, 0.15.0):** v15 + cascade 3 bước tinh chỉnh box + điểm × độ khít dự đoán. [docs/OPENOBB16.md](docs/OPENOBB16.md).
  Công tắc mới `v14_off=stage4,widehead` cho bản nhanh. Chờ log v15 / v16 của user.
- [x] **Chẩn đoán gốc (F64, 10-14):** thước Wall_Color ±0.024; phần thiếu là hình học box, lớn nhất là hai đầu (+0.116); nhãn là quy ước; đầu object trùng
  cạnh object kề. Họ v14–v16 kém v12 bất kể neck (F65).
- [x] **OpenOBB17 (10-15, F66, 0.16.0):** v12 + trường ranh giới stride 2 + bám ranh giới trước NMS. [docs/OPENOBB17.md](docs/OPENOBB17.md).
  Vòng hai hạt giống: fine-tune 30 ep từ `openobb12x.pt`, `openobb17x` vs `openobb12x`, cùng phiên, `nbs=0 warmup_bias_lr=0 epoch_lr=False`.
  Cùng đợt: trainer khớp YOLO (warmup theo epoch tối thiểu 100 iteration, bias LR 0.1, `lf(epoch)`, `nbs=64`).
- [x] **OpenOBB18 (10-15, F67, 0.17.0):** v17 với neck RepConv tự viết thay encoder lai + LSK, chưng cất từ encoder v12 đã train (`teacher=`).
  520 so với 686 GFLOPs, 50.6M tham số. [docs/OPENOBB18.md](docs/OPENOBB18.md). Luật user: mỗi vòng hai version (tinh chỉnh + đột phá) train song song.
- [x] **OpenOBB19 (10-15, F72, 0.18.0):** v18 + head theo nhóm tranh chấp (mỗi điểm lưới một box cho mỗi nhóm class, nhóm tự chia từ nhãn).
  Phát hiện: 60% junction và 86% tường ngắn không có điểm lưới riêng; recall tường < 12 px 0.33 so với YOLO 0.69. [docs/OPENOBB19.md](docs/OPENOBB19.md).
- [x] **Vòng mAP50-95 (10-15, F77, 0.19.0):** v19 + chưng cất box (LD, `teacher=` v17 + nhóm) và OpenOBB20 (v19 + bộ tinh chỉnh cạnh đọc thân stride 4).
  Kết quả trước đó: v17 + nhóm 0.751 / 0.481 (NMS 0.5, trùng 2.2%), v19 0.755 / 0.462, 12.4 ms, trùng 2.8% (F74–F76). [docs/OPENOBB20.md](docs/OPENOBB20.md).
  **Lần 1 bản 1 (F78) lệch thiết kế:** file `openobb19x.pt` của user là v17 + nhóm (neck v19 học lại từ đầu), thầy `openobb17x_v2.pt` là một nhóm
  → 0.752 / 0.464, 12.4 ms, trùng 2.9%, wall → nền 13%.
  **Bản 2 = OpenOBB20-x (F79)**, cùng xuất phát lệch, khác đúng ở bộ tinh chỉnh: **0.764 / 0.482, 12.6 ms, trùng 2.6%**
  (+0.012 / +0.018; wall 50-95 +.028). Bộ tinh chỉnh giữ lại.
  **Nhưng trên bản vẽ thật (F82) box của v20 nghiêng / lệch nặng ở tường dài và tường chéo; nguyên nhân là neck nhanh của họ v18 / v19
  (tắt bộ tinh chỉnh còn tệ hơn).** v17 + nhóm (F74: 0.757 / 0.487 ở o2m 0.7; 0.751 / 0.481 ở o2m 0.5, 16.1 ms) ngang v20 về mAP và đúng
  hướng box → **bản tốt nhất thực tế vẫn là v17 + nhóm**. Thước mới: `research/analysis/infer_box_eval.py` (đuôi sai số góc / đầu box theo độ dài).
- [x] **Lỗi lệch góc + OpenOBB21 (10-15, F83–F84, 0.20.0):** mọi model dense lệch góc ở tường DỌC (88% so với 0.3% tường ngang; YOLO 0%),
  do loss góc 1 − cos 2Δθ (không có gradient gần tối ưu) + lớp ra bf16 ở ±π/2. Sửa chung bật mặc định: loss góc vectơ (cos 2θ, sin 2θ),
  lớp ra hồi quy fp32; thước `Box geometry` in trong mọi val. v21 = v20 + nhánh sửa góc góc kép. [docs/OPENOBB21.md](docs/OPENOBB21.md).
  Checkpoint cũ phải fine-tune lại bằng code mới. Vòng này: A `openobb20x` vs B `openobb21x`, cùng `weights=openobb20x_v2.pt`.
  **Nguyên nhân gốc (F86, 0.20.1):** mẫu dương tường dọc dự đoán ngang (góc khởi tạo 0) → trọng số TAL ~0 → không được dạy góc. Sửa: trọng số
  loss góc đều (`--angle-balance uniform`, mặc định). F85 (v17, 0.20.0, còn `rotate_p=0.5`, nbs 64) vẫn lệch 24%: cần 0.20.1 + bỏ rotate_p + nbs=0.
- [x] **YOLO26x cùng thước (F89–F90):** v21 .769 / .492, trùng 1.8%, 12.7 ms so với YOLO .743 / .499, 1.6%, 7.3 ms (máy nóng); hướng box
  14.4% so với 0.3%. **OpenOBB22 (F91, 0.21.0):** v21 + góc sigmoid 45° + hồi quy box thẳng. [docs/OPENOBB22.md](docs/OPENOBB22.md).
  Cặp đang chờ: A = v21 không `rotate_p` (user đang chạy), B = `openobb22x epochs=150` từ best.pt v21.
- [x] **v22 thất bại (F93):** hồi quy thẳng làm wall sụp (.466). **Profile G4 (F94):** thân + neck 7.5 ms, NMS ~3 ms. **F95:** trùng của head một-một
  là điểm neo kề nhau, nhưng lọc đỉnh mất 0.035 mAP50 (không mặc định); head một-một nghiêng ít hơn (5.5% / 15.4%) → hiệu chỉnh chọn theo độ nghiêng.
  **OpenOBB23 (0.22.0):** v21 + bám ranh giới có xoay. [docs/OPENOBB23.md](docs/OPENOBB23.md). Lệnh: `model=openobb23x weights=<best.pt v21> epochs=100`.
  Không đề xuất đổi thầy nữa (v19 qua ba kiểu thầy, tổng không đổi; user phản biện). Vòng sau chưa chốt.
- [x] **Mốc hiện tại (F107, 10-16): OpenOBB23-x 0.798 / 0.526, trùng 1.8%, 7.9 + 1.3 ms** (fine-tune nối tiếp F96 → F103 → F107, thân đóng
  băng, khung YOLO 0.23.0) so với YOLO26x 0.743 / 0.499. Còn thua ở wall / junction / wall_300 (độ khít box) và hướng box vật dài
  (6.0% box dài lệch trục so với 0.3%). Train tiếp gần bão hoà trên 250 ảnh (F102–F107). Mở khoá thân thua đóng băng (F100–F106).
- [x] **OpenOBB24 (F109, 0.25.0, chưa train):** chẩn đoán trên checkpoint F107: recall đã ngang YOLO, khoảng hụt là độ khít box; điểm class
  không biết độ khít, trong cụm NMS có box khít hơn (+0.052), tín hiệu trường ranh giới thô chọn box trong cụm đã +0.015 không cần train.
  v24 = v23 + `LocQuality` (dự đoán ProbIoU của box cuối) + NMS theo điểm × q^β + bỏ bám mép theo nhóm, chọn trên val.
  [docs/OPENOBB24.md](docs/OPENOBB24.md). Mốc so là F107 (user 10-19: không chạy đối chứng v23 riêng); user đang train `openobb24x` 50 ep, lr 0.001, từ `openobb23x_new.pt`.
  **Kết quả (F110): 0.802 / 0.531, đầu độ khít học được bị bác bỏ** (mọi q^β kém base 0.5564, hiệu chỉnh chọn q^0); bỏ bám mép nhóm
  note + junction được giữ (junction .559, +.017). Bước tiếp: `quality_probe.py` trên best.pt v24, thử tín hiệu trường thô làm lựa chọn.
  **F111:** q bị bác bỏ do hai lỗi đường suy luận (ứng viên chưa tinh chỉnh nhận q = 0.6; ngưỡng conf trên điểm × q^β), sửa ở 0.25.2;
  sửa xong q chỉ +0.002–0.003 fitness → chọn lại box trong cụm NMS không phải lối ra cho độ khít box.
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

## 8b. Quy trình nghiên cứu (handoff, chuẩn từ 2026-10-14)

Agent (Claude Code, Codex, Cursor...) ở session khác làm theo đúng thứ tự này. Không có GPU local; user train trên Colab và gửi log.

### B1. Đọc trước khi làm gì
1. File này: mục 7 (quyết định của user), mục 8 (trạng thái), mục 8b, 3–5 mục cuối của Nhật ký (mục 9).
2. [research/LEDGER.md](research/LEDGER.md): các F mới nhất (hiện F58–F62). Mọi số liệu đều ở đó, không hỏi lại user số đã ghi.
3. Thẻ kiến trúc của 2–3 version mới nhất: `docs/OPENOBB14.md`, `OPENOBB15.md`, `OPENOBB16.md` (mỗi thẻ có bảng lập luận và bảng kỳ vọng ghi trước).
4. Bài học tổng hợp: [research/LESSONS_BY_LAYER.md](research/LESSONS_BY_LAYER.md), [research/ARCH_ANALYSIS.md](research/ARCH_ANALYSIS.md).

### B2. Luật thiết kế (của user, không thương lượng)
- **Mốc so sánh:** YOLO26x-obb trên Wall_Color **0.746 / 0.499** (F59) và bản OpenOBB tốt nhất hiện tại **v12-x head một-nhiều + NMS 0.7:
  0.743 / 0.472, 15.3 ms / ảnh (G4, batch 16)**. So cùng thước YOLO (`metric=yolo`, in trong log), cùng data, cùng cách hiệu chỉnh head / NMS.
- **Mục tiêu:** chính xác hơn YOLO (nhất là mAP50-95); chậm hơn YOLO được nhưng không quá (khoảng ≤ 1.3× v12 là trần tạm). Kiến trúc và weights
  phải **thương mại được, thuộc sở hữu user**: chỉ thành phần Apache-2.0 / MIT, không code / weights Ultralytics (`tests/test_license_guard.py`
  chặn import). Bản mới na ná kiến trúc cấm thương mại thì vô nghĩa.
- **Version mới = đột phá có lập luận.** Mỗi `openobbN` phải có thẻ `docs/OPENOBBN.md` gồm: bằng chứng (số F), bảng *bài học → cơ chế → kỳ vọng →
  điều chứng minh sai*, chi tiết từng phần + tỉ lệ tận dụng phần có sẵn, **bảng chỉ số kỳ vọng ghi TRƯỚC khi train** và tiêu chí bác bỏ.
  Tinh chỉnh nhỏ thì thành công tắc (`v14_off=`, `box=`, `neck=`...), không thành version.
- **Mỗi phần mới khởi tạo đồng nhất** (zero-init) trên bản trước, có test "bắt đầu đúng bằng bản trước" (ví dụ `test_v16_starts_as_v15`).
- **Mỗi vòng giao user HAI hạt giống** khác nhau đúng một yếu tố, kèm lệnh `openobb train ...` chính xác; user train song song và gửi log.
- **Đào sâu bên trong mọi backbone SOTA bằng số đo**, không lắp thêm mù: trước khi thiết kế, chạy các script ở B3 và trích số vào thẻ.
- Giữ thân HGNetv2 pretrained cho tới khi có ngân sách pretrain thân riêng (3 lần train lõi từ đầu đều thất bại: F25, F40, F50). Lộ trình sở hữu
  trọn: thân conv dày tái tham số hoá, chưng cất từ chính model đã train trên bản vẽ + DOTA.

### B3. Công cụ đo (chạy local, CPU, không cần GPU)
| Việc | Lệnh |
|---|---|
| Mổ cơ chế bên trong thân trên checkpoint đã train | `OMP_NUM_THREADS=2 py -3.12 research/analysis/dissect_hgnet.py F:/Downloads/openobb12x.pt --images 3` |
| FLOPs / byte / roofline G4 từng mảnh (model của mình và YOLO) | `OMP_NUM_THREADS=1 py -3.12 research/analysis/roofline.py openobb16x:neck=dfine openobb12x yolo26x-obb --hw 960x1280` (biến thể: `openobb15x:neck=dfine,off=stage4,widehead`) |
| 16 backbone SOTA cùng thước | `py -3.12 research/analysis/roofline_backbones.py` (cần `timm`, chỉ để đo) |
| Độ khít box so với YOLO trên checkpoint | `research/analysis/box_quality.py`, `dup_analysis.py` (xem đầu file) |
| Test | `OMP_NUM_THREADS=1 py -3.12 -m pytest -q tests/` (khoảng 4 phút; torch CPU trên máy này phải 1 luồng) |

Checkpoint đã train: `F:\Downloads\openobb12x.pt`, `openobb10x.pt`; lõi DOTA: `runs/e6-vrdet-x-dota-24e/openobb_dota_core_x.pt`.
Data Wall_Color: `datasets/dataset_wall_color` (gitignored).

### B4. Thêm một version N (danh sách kiểm, làm đủ)
1. `openobb/models/openobbN.py` (+ `openobbN_loss.py` nếu cần). Kế thừa bản trước; mọi phần mới zero-init; `fuse()` gộp BN / RepConv.
2. Đăng ký: `openobb/cli.py` (`MODELS`, `RECIPEN`, `SIZE_DEFAULTSN`, tuple `dense_arch`, map `sd` / recipe, `ARCH_KEYS` nếu có cờ mới, nhánh
   `imgsz` 2× nếu ảnh vào gấp đôi như v13 / v15 / v16); `openobb/predict.py` (`KNOWN_ARCHS`, nhánh nạp, `input_size`); `openobb/train.py`
   (choices `--arch`, nhóm `v3`, dựng model, loss, dòng log `model`, copy head một-một khi `weights=`).
3. `tests/test_openobbN.py`: train step + loss hữu hạn + gradient mọi phần mới; đồng nhất lúc đầu so với bản trước; fuse giữ đầu ra; nạp weights
   pretrained / checkpoint cũ; suy luận mọi cỡ; registry. Chạy cả `tests/test_strip.py` (mọi model trong `MODELS` phải có trong `KNOWN_ARCHS`).
4. `docs/OPENOBBN.md` (theo B2) + thêm dòng vào bảng model của `README.md`, `docs/FINETUNE.md`, `tools/gitlab_README.md`.
5. Sổ cái: mục F mới (thiết kế + kỳ vọng ghi trước). Mục 8 + Nhật ký của file này.
6. Commit chỉ file của mình (`git add <file>...`; `app/streamlit_app.py`, `docs/DEPLOY.md` và một phần `tools/gitlab_README.md` đang có sửa dở
   của user, không commit). Thông điệp commit kết bằng dòng `Co-Authored-By` theo quy định của session.
7. Phát hành: `python tools/push_all.py` (`--minor` khi có kiến trúc mới): tự tăng version, push GitHub, đồng bộ bản sạch sang GitLab (đổi tên
   `hicasobb`), chạy test trên bản đó, build wheel, gắn tag. Colab của user cài từ GitHub, nên **phải push trước khi bảo user train**.
8. Trả lời user bằng tiếng Việt: hai hạt giống + lệnh + điều cần gửi lại (bảng theo class, confusion matrix, `Duplicate boxes`, `NMS calibrated`,
   `Speed`, và `openobb val model=best.pt data=... ref_k=0` với họ v14+).

### B5. Khi user gửi log
1. Ghi mục F mới: điều kiện (init, epoch, lr, batch, imgsz, GPU), bảng theo class so với mốc (F59) và bản trước, dòng hiệu chỉnh, box trùng, tốc độ.
2. Đối chiếu từng dòng của bảng kỳ vọng trong thẻ; ghi **đạt / không đạt / bác bỏ** vào thẻ (mục "Kết quả thật") và sổ cái. Không nới tiêu chí sau khi có số.
3. Chẩn đoán theo class (class nào mất, class nào tăng) → nguyên nhân có bằng chứng → phép thử tách một yếu tố cho vòng sau (công tắc có sẵn
   trước, version mới sau).
4. Cập nhật mục 8 của file này và memory của agent nếu user ra luật mới.

### B6. Trạng thái vòng hiện tại (2026-10-14 tối)
- Đã có: `openobb15x` neck mặc định (F63, 0.653 / 0.416). Cần chạy: `openobb15x neck=dfine` và `openobb14x neck=dfine`
  (kiểm tra dòng `model` của log có "D-FINE COCO hybrid encoder (control)").
- Sẵn sàng train: `openobb16x neck=dfine` (v16, 0.15.0). Bản nhanh nếu quá chậm: thêm `v14_off=stage4,widehead`.
- Cây quyết định khi có log:
  - v15 ≥ 0.76 / 0.51 và wall_300 ≥ .78 → cơ chế độ phân giải gốc được xác nhận; chạy v16 từ `best.pt` của v15 (`weights=`), so cùng xuất phát.
  - v15 tốt ở 50-95 nhưng chậm quá → `openobb15x neck=dfine v14_off=stage4,widehead` (đo trước bằng roofline).
  - v15 không hơn v14 `neck=dfine` → luồng gốc / P2 không có tác dụng ở 250 ảnh; quay về câu hỏi tiên nghiệm (H-DOTA: `weights=` lõi DOTA,
    hoặc `pretrain=dota`), và kiểm tra lại nhãn wall_300.
  - v14 `neck=dfine` ≥ v12 ở 50-95 mà v14 neck tự thiết kế không → neck mất tiên nghiệm là nguyên nhân F60; lộ trình sở hữu trọn phải qua
    chưng cất neck / thân.
- Chưa làm: pretrain thân riêng (cần CU), FloorPlanCAD cho v14–v16 (lệnh ở `docs/OPENOBB14.md` mục 4c).

### B7. Lỗi môi trường hay gặp
- Bash tool trên Windows: heredoc chứa nhiều dấu nháy đơn làm shell báo "unexpected EOF"; khi đó **ghi script Python ra file trong scratchpad
  (Write tool) rồi chạy**, không nhúng vào heredoc.
- torch CPU trên máy này: luôn `OMP_NUM_THREADS=1` (hoặc 2) để không crash heap; tốc độ CPU local không dùng để so model (F47), chỉ GPU cùng phiên.
- `timm` và `ultralytics` có trong Python 3.12 local chỉ để đo (research), không bao giờ import trong `openobb/`.
- `push_all.py` chỉ đồng bộ file đã commit; nó in danh sách file chưa commit, đọc để không bỏ sót.

## 9. Nhật ký

- 2026-10-21 (2): **F117: `openobb export` (ONNX) + wheel runtime `openobb_onnx` để bàn giao** (theo user: wheel `hicasobb` lộ kiến trúc
  + code train). Bàn giao = `best.onnx` + wheel `hicasobb_onnx` (không PyTorch, không code kiến trúc / train). Model mặc định để giao:
  **OpenOBB23-x** (user: v24 không hơn v23). Khớp `.pt` 1158 / 1158 box; CPU nhanh hơn 2.4×. Tài liệu [docs/ONNX.md](docs/ONNX.md),
  test `tests/test_onnx.py`, build `tools/build_onnx_wheel.py` (push_all gọi). Thư mục thử cài: `F:\Source_code\OpenOBB_test_wheel`
  (`.venv` không torch, `run_app.bat`). Chưa đo GPU.
  Cùng ngày: GitLab `wheels/` chỉ còn wheel ONNX (wheel đầy đủ chỉ ở `dist_hicasobb/`); wheel ONNX cài dưới tên `hicasobb` để code
  deploy cũ chạy nguyên, chỉ đổi path `.pt` → `.onnx`; gói đầy đủ cũng nhận `.onnx`. Phát hành v0.25.8 → v0.25.10.

- 2026-10-21: **F116: `compile=` cho predict / val giống YOLO** (theo user; mặc định tắt). `predict.compile_mode` / `compile_model`
  (compile `encode` + trường ranh giới, head eager; lỗi trình biên dịch → cảnh báo, chạy bản thường), `OpenOBB.predict(compile=)`,
  `openobb predict|val compile=True|<mode>`; `OPENOBB_COMPILE=1` vẫn dùng khi không truyền. Bản sao val trong train luôn eager.
  Test `tests/test_compile.py`. Chưa đo GPU. Repo đổi tên thư mục `NEW_architecture` → `OpenOBB` (10-20).

- 2026-10-19 (8): **F115 (0.25.6):** val mỗi epoch chỉ thước YOLO (theo user) + model gộp BN (`train.fused_copy`); val cuối giữ
  đủ. Warmup giữ 3 epoch như YOLO (user chốt). Chờ: thời gian val mỗi epoch mới (YOLO 11 s, 0.25.4 44 s).

- 2026-10-19 (7): **F114 (0.25.5):** user đo `compile=True` trên 0.25.4: 18.6 → 12.6 → 9.5 s / epoch (dataset nhỏ; tổng −49%).
  `compile=True` thành mặc định cho v18–v24 (`compile=False` để tắt). Chờ: mAP có compile so với không.

- 2026-10-19 (6): **F113 (0.25.4):** user đo 14.5 → 9.5 phút / epoch sau F112 (YOLO26x 8.5). FLOPs forward thấp hơn YOLO
  (428 so với 507, `research/analysis/train_flops.py`); phần chậm còn lại là hiệu suất thân: `FrozenBatchNorm2d` dùng kernel
  `F.batch_norm` (giữ bf16), `compile=True` giờ compile thân + neck + field + head (`train.compile_training`), checkpoint ghi nền
  (`AsyncSaver`). Profile: `openobb profile model=... train=True [legacy=True] [compile=True]`.

- 2026-10-19 (5): **Tốc độ train (F112, 0.25.3):** gán nhãn dày dạng batch (`assign_batched`), đích trường vẽ theo cửa sổ, loss bộ
  tinh chỉnh không lặp ảnh; kết quả giữ nguyên (test). `openobb profile model=... train=True` đo từng phần một bước train trên GPU.
  Chờ user: log profile `assign=batched` / `assign=legacy` + thời gian epoch mới.

- 2026-10-19 (4): Hiệu chỉnh v24 thử thêm `q_outside` `min` / `drop` và lưu vào `best.pt` (0.25.2). Lệnh giao user: hiệu chỉnh lại
  best.pt F110 + train lại v24 không `rotate_p` (hướng box F110 xấu hơn F107: box dài lệch trục 8.5% so với 6.0%).

- 2026-10-19 (3): Đóng việc dở của `docs/HANDOFF_OPENOBB_F111_2026-10-19.md`: lượt 53 ảnh / 6 cấu hình đã xong trước khi phiên cũ dừng
  (`research/analysis/quality_deployed_v24f110.log`), số khớp F111; nguồn số ghi thêm vào F111. Test v24 + strip chạy lại: pass.

- 2026-10-19 (2): **F111: hai lỗi đường suy luận của v24 (0.25.2).** `q_outside="min"` (mặc định) / `"drop"`; `fast_nms` lọc conf theo
  điểm xuất ra. Script `research/analysis/quality_deployed.py` (đường suy luận thật), `quality_probe.py` đọc q học được.

- 2026-10-19: **Log OpenOBB24-x (F110): 0.802 / 0.531.** Đầu độ khít học được không giúp (hiệu chỉnh chọn q^0); bỏ bám mép cho
  note + junction giữ lại. Kết quả ghi ở `docs/OPENOBB24.md` mục 6.

- 2026-10-16 (7): **OpenOBB24 + chẩn đoán F109** (theo user: tìm phần còn nhiều không gian tối ưu nhất).
  Script đo mới: `rank_vote_oracle.py`, `score_iou_corr.py`, `consensus_probe.py`, `band_width_probe.py`, `quality_probe.py`;
  `dup_analysis.py DUP_STAGES=head|edge`. `fast_nms(out_logits=)`, `EdgeRefiner.refine_with_fit(hidden=)`, `fit_loss` đọc `ref_fit_valid`.
  Sửa `strip` thiếu `groups` (checkpoint v19+ đã strip không nạp được).

- 2026-10-16 (6): **0.24.0: LR giảm theo cosine là mặc định** (theo user, F108). `cos_lr=False` để về lịch tuyến tính của YOLO.
  Đánh giá theo đồ thị LR (F108): hợp lý; user chốt không cần chạy A/B riêng.

- 2026-10-16 (5): **Log v23 fine-tune thêm 30 epoch (F107): 0.798 / 0.526**, mốc mới. README hai repo, `docs/OPENOBB23.md` (bảng các lần
  fine-tune) đã cập nhật. Bản GitLab có thêm thẻ v14–v23. App thử model: viewer toàn màn hình (`app/viewer/index.html`).

- 2026-10-16 (4): **0.23.0: khung training YOLO là mặc định duy nhất** (theo user, F104). Bỏ `trainer=` (truyền vào thì bỏ qua) và
  `_auto_recipe`; `FRAMEWORK` trong `openobb/cli.py`. `unlock_all` giữ như tính năng kiến trúc. Test, FINETUNE, README hai repo đã sửa.

- 2026-10-16 (3): **Tách khung training và kiến trúc model** (F99, theo user). Rà Ultralytics 8.4.174; thêm `trainer=yolo` (chỉ khung:
  LR theo số class, wd 5e-4, LR tuyến tính, EMA 0.9999/2000, augment, không RFS, nbs 64) và `unlock_all=True` (kiến trúc: không đóng băng gì).
  Mặc định không đổi; chờ log A/B/C của user trên Wall_Color.

- 2026-10-16 (2): **Bộ khung train khớp lại YOLO hiện tại + tự chọn công thức** (F98, log fine-tune bộ tivi 77 ảnh của user).
  0.22.4: `warmup_bias_lr=0` (YOLO đặt 0 khi dùng AdamW; 0.1 làm fine-tune tụt). 0.22.5: `optimizer=auto` tự đặt `nbs=0` khi < 2000 bước
  và `lr0 = max(mặc định, 0.002·5/(4+nc))` khi có class mới / không có weights. 0.22.6: warmup thuần theo epoch, bỏ mức tối thiểu
  100 iteration (Ultralytics 8.4 đã bỏ). Chưa đo trên GPU; chờ log user.

- 2026-10-16: **Phụ lục "kiến trúc theo lớp" cho mọi thẻ `docs/OPENOBB1–23.md`** (theo user). Mỗi thẻ có mục cuối: biểu đồ Mermaid
  (xanh dương mượn, xanh lá tự viết, cam mới ở version đó) + bảng từng khối (lớp, kênh, kernel, kích thước ra ở ảnh 1280, tham số) + tổng hợp
  tỉ lệ tự viết. Sơ đồ ASCII cũ trong thân thẻ thay bằng dòng trỏ tới phụ lục. Số liệu đo bằng `research/analysis/arch_layers.py`
  (dựng model, đếm tham số, shape từng khối; `OMP_NUM_THREADS=1 py -3.12 research/analysis/arch_layers.py v23 --depth 3`).
  Thẻ version mới phải có mục này (ở cuối file). README GitHub / GitLab: bảng các kiến trúc đã train kèm kết quả.

- 2026-10-15 (5): **OpenOBB22** (F91), YOLO26x đo cùng thước trên G4 (F89, cell so sánh), `Speed` tách NMS của v10+ (0.20.2).
  Notebook của user truyền `rotate_p=0.5` (mặc định code = 0, giống `degrees=0` của YOLO): nhắc user bỏ.

- 2026-10-15 (4): **Đánh giá hướng box + sửa lỗi góc + OpenOBB21** (F82–F84). User thấy v20 lệch hướng trên bản vẽ thật; thước mAP không thấy.
  `research/analysis/eval_versions.py`, `tilt_real.py`, `tilt_probe.py`; `openobb.eval.ultra.geometry`; `--angle-loss`; `openobb3.reg_out`; `OpenOBB21`.
  Dòng `augment` giờ in `rotate any angle` (trước đây ẩn) và `angle loss`.

- 2026-10-15 (3): **Log OpenOBB20-x** (F79): 0.764 / 0.482, 12.6 ms. Cặp với F78 khác đúng một yếu tố (bộ tinh chỉnh), nên giữ v20.
  Đề xuất đổi thầy cho vòng sau đã bỏ theo phản biện của user (F79).

- 2026-10-15 (2): **Log bản 1 vòng F77** (F78). Mở checkpoint local thấy `openobb19x.pt` = v17 + nhóm, `openobb17x_v2.pt` = v17 một nhóm
  (nạp thử vào v19-x khớp đúng `1084 / 12 / 328`, 328 tensor mới là neck). Kết quả 0.752 / 0.464 không dùng để chấm kỳ vọng.
  Giả thuyết: KD class từ thầy một nhóm kéo nhóm wall xuống (wall → nền 13%). Kiểm tra dòng `[init]` trước mỗi lần train.

- 2026-10-15: **OpenOBB17 + trainer khớp YOLO** (F64–F66).
  - User: đừng quanh quẩn tinh chỉnh nhẹ; warmup phải giống YOLO. Đo F64 (CPU) → hướng "detector theo ranh giới"; log v14 / v15 `neck=dfine`,
    v16 cho thấy họ v14 hỏng bất kể neck → v17 xây trên v12.
  - `openobb/models/openobb17.py`, `openobb17_loss.py`, `tests/test_openobb17.py`; `engine.yolo_lf`, `param_groups(split_bias=)`; trainer `--epoch-lr`,
    `--warmup-bias-lr`, `--nbs`; CLI mặc định `epoch_lr=True, warmup_bias_lr=0.1, nbs=64`, warmup `max(3 epoch, 100)` iteration.
  - `tools/gitlab_README.md` chưa thêm dòng v17 (file có sửa dở của user).

- 2026-10-14 (tối): **Log v15 neck mặc định** (F63): 0.653 / 0.416, luồng gốc + P2 không giúp wall / junction / wall_300, note sụp.
  Kết quả ghi ở `docs/OPENOBB15.md` mục 6a. Bước tiếp vẫn là cặp `neck=dfine` (v15 vs v14).

- 2026-10-14: **OpenOBB14 + mổ bên trong backbone** (F58, [docs/OPENOBB14.md](docs/OPENOBB14.md); chưa train).
  - User: phải đào sâu vào bên trong MỌI backbone SOTA (không chỉ lắp thêm lên HGNetv2) mới có đột phá; mốc so sánh là
    YOLO26x và v12 tốt nhất = head một-nhiều + NMS hiệu chỉnh (0.741 / 0.481, F56), không phải `box=direct`.
  - Đo mới (CPU, không cần GPU):
    - `research/analysis/dissect_hgnet.py`: can thiệp từng cơ chế trong thân HGNetv2 của `openobb12x.pt` đã train.
      Thân gần như toàn bộ sống trên bản vẽ; depthwise 5×5 ở stage 3 thiết yếu (cắt còn 3×3 mất 38% detection);
      stage 4 chỉ góp 4%, 23% kênh chết; B4 / B5 không có ESE.
    - `research/analysis/roofline.py`: FLOPs + byte + ước lượng roofline G4 theo từng mảnh (v12-x, v14-x, YOLO26x).
    - `research/analysis/roofline_backbones.py`: cùng thước cho 16 backbone SOTA (timm + Ultralytics, chỉ để đo).
  - v14 = thân HGNetv2 giữ nguyên + neck RepCSP tự thiết kế (thay LSK + encoder lai) + head kép nặng hồi quy trực tiếp
    + bộ tinh chỉnh box đọc nét stride 4 trên cạnh box. 62.7M (47% tự viết), 586 GFLOPs so với v12 682 / YOLO26x 605.
  - `model=openobb14s|m|l|x`, `v14_off=`, `box=direct|dfl`, `ref_k=`. Test `tests/test_openobb14.py`.
  - Cổng trước khi tốn CU: T1 tốc độ G4 (`colab/speed_gpu.py`) ≤ 0.85× v12.
  - Đã push: GitHub v0.13.0 (081282d), GitLab hicasobb v0.13.0 + wheel. User chỉ cần đổi `model=openobb14x` trên Colab.
  - User nhắc (10-14): mục đích là kiến trúc **thương mại được, sở hữu được**, chính xác và nhanh hơn; bản mới mà na ná
    kiến trúc cấm thương mại (YOLO) thì vô nghĩa. Trạng thái v14: code 100% tự viết, không có code / weights Ultralytics;
    phần mượn duy nhất là thân HGNetv2 + weights D-FINE COCO (Apache-2.0, dùng thương mại được). Lộ trình để sở hữu
    trọn: v15 = thân conv dày tái tham số hoá (cường độ 483 FLOP/byte, F58) pretrain bằng chưng cất từ chính v14 đã train
    trên bản vẽ + DOTA (`pretrain=dota`, khoảng 25–40 CU), thay cho thầy COCO đã thất bại ở v9 (F50).

- 2026-10-13 (9): **0.12.0: head suy luận hiệu chỉnh, `box=direct`, `openobb calibrate`, sửa `imgsz` của v13** (F54, F55).
  - v12 (user): 0.722 / 0.468. Cùng checkpoint, head một-nhiều + NMS 0.5 cho 0.741 / 0.481.
    → `args["infer_head"]` được hiệu chỉnh cuối train (`predict.postprocess_options`, `apply_head`); lệnh `openobb calibrate`.
  - Đo độ khít box so với YOLO26x (`research/analysis/box_quality.py`): box dày hơn thật, đầu box và góc kém.
    → `box=direct` (kiểu YOLO26, `openobb3.DIRECT_CH`, `openobb3_loss.dist_loss`).
  - v13: `imgsz` giờ là cỡ của lõi (`predict.input_size`). Run v13 đầu tiên của user không hợp lệ (lõi 640).

- 2026-10-13 (8): **OpenOBB13 thay bằng bản nhắm độ chính xác** (0.11.0, F52, [docs/OPENOBB13.md](docs/OPENOBB13.md)).
  - User bỏ v13 cũ (lõi nhanh v9) vì chỉ nhắm tốc độ.
  - Đo trên nhãn: 53% wall_300 chung ô P3 với junction. v13 = v12 + luồng độ phân giải gốc (`hires`) + lưới P2 (`p2_head`, `p2_head_o2o`) + `to_p3`.
  - `imgsz` của v13 là độ phân giải đầy đủ (Wall_Color: 2560), lõi chạy ở một nửa. Model tự đệm đầu vào lên bội 64.
  - Checkpoint v13 của 0.10.0 không nạp được nữa.

- 2026-10-13 (7): **OpenOBB12 / OpenOBB13: head dense kép, không NMS** (F51, [docs/OPENOBB12.md](docs/OPENOBB12.md)).
  - `openobb/models/openobb12.py` (`DualHeadMixin`, `OpenOBB12` lõi v7, `OpenOBB13` lõi v9) + `openobb12_loss.py` (loss dense v10 cho cả hai head, head một-một gán k = 1).
  - `pretrain_teacher=` nhận detector OpenOBB đã train (`pretrain.TRAINED_CORES`): thầy là `encode()` của nó. `pretrain=` dùng được cho v13.
  - Cột `o2o_loss` trong báo cáo. Test: `tests/test_openobb12.py`.

- 2026-10-13 (6): **Tham số suy luận do train quyết định** (theo user: "NMS phải là tham số training, không hardcode").
  - `--nms-iou` (CLI `iou=` / `nms_iou=`, mặc định `auto`): train xong thử các ngưỡng trên val, lưu tốt nhất vào `best.pt`.
  - `infer_k`: số ứng viên nhỏ nhất của tầng quan hệ giữ nguyên fitness.
  - `predict.apply_nms` / `merge_iou_for` / `load_model` áp các giá trị này; `strip` giữ lại.

- 2026-10-13 (5): **OpenOBB11 + wheel sản phẩm** (F49, [docs/OPENOBB11.md](docs/OPENOBB11.md)).
  - User giao toàn quyền làm v11. Phân tích box trùng của v10 (53 ảnh val): gốc ở kiến trúc → tầng quan hệ train một-một, suy luận không NMS.
  - `openobb8_loss.ref_assign` (tách ra, hành vi v8 không đổi); `OpenOBB8._with_jitter(return_idx=True)`.
  - `push_all.py` build wheel `hicasobb-X.Y.Z-py3-none-any.whl` (khoảng 200 KB) mỗi lần phát hành:
    - kiểm tra rò rỉ bên trong wheel;
    - đẩy vào `wheels/` của GitLab (chỉ giữ bản mới nhất);
    - lưu một bản ở `dist_hicasobb/` local (gitignored).

    Khách cài bằng `pip install <wheel>`, không cần quyền repo.

- 2026-10-13 (4): **0.8.2: kiểm tra kiến trúc khi nạp model.**
  - Quy trình user chốt: mọi thay đổi lên cả hai repo cùng phiên bản. GitHub `openobb` dùng cho nghiên cứu và train trên Colab;
    GitLab `hicasobb` (project `.../openanything/hicasobb`, lịch sử sạch, đã force-push một lần theo lời user) chỉ dùng để nạp model và suy luận.
    Project GitLab `openobb` cũ bị bỏ, user tự xoá.
  - Checkpoint ghi `version` (bản lúc train).
  - `predict.check_arch`:
    - nhãn `arch` không có trong `KNOWN_ARCHS` → báo lỗi rõ, kèm yêu cầu cập nhật (trước đây rơi xuống dựng v1 và lỗi lệch tensor);
    - model train ở bản mới hơn → cảnh báo.
  - **Khi thêm kiến trúc mới phải thêm nhãn vào `KNOWN_ARCHS`** (test `test_check_arch_unknown_and_newer` bắt lỗi nếu quên).

- 2026-10-13 (3): **Repo sản phẩm chuyển sang GitLab `.../openanything/hicasobb`, lịch sử mới; `openobb strip`** (0.8.1).
  - User chọn project mới để bỏ lịch sử cũ (project `openobb` cũ chứa package `openobb` và link repo GitHub; user tự archive).
  - `push_all.py`:
    - đổi URL;
    - tự clone lại khi remote đổi, hỗ trợ project trống;
    - commit message trung tính `Release vX.Y.Z`;
    - xoá mọi nhắc tới repo nghiên cứu, `research/`, `vrdet` trong bản sản phẩm;
    - **chặn đẩy** nếu còn `openobb|vrdet|nguyenchitrai|research/|repo nghiên cứu`.
  - `openobb strip model=best.pt`: file giao khách chỉ còn weights + kiến trúc + class + ngưỡng conf
    (bỏ đường dẫn `/root/.cache/openobb/...`, tham số train, điểm số), tự kiểm tra không còn tên nội bộ.

- 2026-10-13 (2): **0.8.0: bản GitLab đổi thương hiệu thành HicasOBB** (theo user; GitHub giữ `openobb`).
  - `tools/push_all.py` tự đổi khi đồng bộ: package `openobb` → `hicasobb` (từ 0.8.3; 0.8.0–0.8.2 là `hicas_detection_obb`), class `OpenOBB*` → `HicasOBB*`,
    model / module `openobbN` → `hicasobbN` (`model=hicasobb8x`), lệnh CLI `openobb` → `hicasobb`, `OPENOBB*` → `HICASOBB*`.
    URL repo giữ nguyên. Test chạy trên bản đã đổi tên trước khi đẩy.
  - Quy trình user: train trên Colab bằng GitHub (`openobb`), suy luận bằng GitLab (`hicasobb`).
    Đã kiểm: checkpoint chỉ chứa dữ liệu (`torch.load(weights_only=True)` được), `openobb10x.pt` nạp bằng bản GitLab cho
    đầu ra giống hệt bản GitHub. Cần dùng cùng số phiên bản ở hai bên.

- 2026-10-13: **`openobb train ... pretrain=coco|dota|<thư mục>`** (theo user: pretrain v9 bằng một lệnh, khung tự lo data).
  - `openobb/auto_pretrain.py` + `openobb/data/download.py`:
    - tự tải COCO val2017 / DOTA-v1.0 (cắt 1024);
    - chưng cất lõi (thầy D-FINE COCO hoặc `pretrain_teacher=<.pt>` + DINOv2);
    - với `dota` thì train có nhãn trên DOTA;
    - cache theo cài đặt (`pretrain_dir=`).
  - `openobb pretrain teacher_dfine=<.pt>` nhận lõi DOTA làm thầy (0.7.2; F48).
  - User bác bỏ `openobb11` dạng gộp v9 + cân bằng class + bỏ tầng quan hệ (đã gỡ, không commit).
    Luật: version mới phải là cải tiến đột phá, không gói tinh chỉnh nhỏ.

- 2026-10-12 (5): **OpenOBB10 + suy luận ảnh chữ nhật** (F45, [docs/OPENOBB10.md](docs/OPENOBB10.md)).
  - Đo tốc độ trên trang thật (CPU 1 luồng): v3-x 1.32×, v8-x 1.77× YOLO26x cùng đầu vào. Trên trang A, OpenOBB còn đệm vuông
    (thêm 38% điểm ảnh) → suy luận chữ nhật (bội số 32) thành mặc định cho mọi model, không cần train lại.
  - Đo trên nhãn: 72% wall_300 không có điểm lưới riêng (tranh với junction) → head 3 slot / điểm;
    78% điểm P3 của note học box cụt → gán nhãn theo tầm với; RFS không tác dụng với class hiếm → cân bằng mức object.
  - User: không cắt nhánh P5, không thay phần pretrained (rủi ro độ chính xác); không train trên CPU local; không mặc định giao sub-agent.
  - `model=openobb10s|m|l|x`, `v10_off=slots|reach|balance`. Test: `tests/test_openobb10.py`.
  - F46: đặt P5 = 0 trên trang thật: v3 không đổi (P5 chết), v8 giữ 80.7% box (P5 sống nhờ ScaleGuard) → giữ P5.
  - 0.7.1: LR mặc định giảm theo cosine (`cos_lr=True`), theo user.

- 2026-10-12 (4): **Sửa tốc độ suy luận cho họ v3, thêm OpenOBB9** (F42).
  - Tốc độ: gộp BN khi suy luận + lọc `conf` trước NMS, áp dụng cho v3 / v5 / v7 / v8; CPU −18%, đầu ra không đổi.
  - OpenOBB9: lõi mới tự thiết kế (conv dày tái tham số hoá), 543 GFLOPs so với 969 của v8, cùng head với v8.
    Pretrain chưng cất 2 thầy: `openobb pretrain`, [docs/PRETRAIN.md](docs/PRETRAIN.md).
  - Làm song song bằng 2 sub-agent (kiến trúc / pretrain). 174 test pass.
  - User yêu cầu: mọi bản mới phải qua đo tốc độ GPU (T1) trước khi tốn CU train.

- 2026-10-12 (4): **Bài học theo lớp model** ([research/LESSONS_BY_LAYER.md](research/LESSONS_BY_LAYER.md)) tổng hợp v1–v8 so với D-FINE và SOTA.
  Log v8 đang chạy ghi ở LEDGER F42 (ep37 0.607 / 0.375 thước YOLO, hội tụ nhanh nhất từ trước tới nay).

- 2026-10-12 (3): **OpenOBB8: bản tốt nhất hiện tại**
  (F41, [docs/OPENOBB8.md](docs/OPENOBB8.md)). OpenOBB8 = lõi OpenOBB7 (v3 đã sửa lỗi, [docs/OPENOBB7.md](docs/OPENOBB7.md)) + tầng suy luận quan hệ.
  - Viết song song bằng 2 sub-agent.
  - `model=openobb8s|m|l|x`; không có tên `openobb7*` (lõi = `v8_off=refine`).
  - **Luật mới của user:** mọi version mới phải có lập luận (bài học cũ → cơ chế → kỳ vọng → điều chứng minh sai), không thử mù.
    Bản nào lý thuyết chứa trọn bản kia thì gộp, không test cả hai.
  - v6 thất bại (F39–F40, khoảng 0.55 / 0.32 khi train từ đầu) → v6 chỉ còn là thử nghiệm.
  - Bản đột phá tiếp theo là **v9**: kiến trúc mới, tự pretrain bằng chưng cất 2 thầy (DINOv2 + lõi D-FINE COCO), lõi conv tái tham số hoá cho GPU.
    Phải viết lập luận trước khi code.

- 2026-10-12 (2): **OpenOBB6: kiến trúc tổng quát tự thiết kế** (F38, [docs/OPENOBB6.md](docs/OPENOBB6.md)).
  - Code:
    - `openobb/models/openobb6.py`;
    - lệnh pretrain chưng cất `openobb/pretrain.py` (`openobb pretrain`);
    - công tắc ablation `v6_off`;
    - `fuse()` gộp BN khi suy luận;
    - notebook `colab/OpenOBB6_wall_color.ipynb`.
  - `model=openobb6n|s|m|l|x`; train với `weights=<file pretrain>`.
  - Test: `tests/test_openobb6.py`. Toàn bộ 117 test pass.
  - Data Wall_Color có sẵn ở `datasets/dataset_wall_color` (250 train / 53 val, gitignored).
  - User: Wall_Color chỉ là bài test nhỏ; kiến trúc phải tổng quát cho detection.

- 2026-10-12: **Phân tích từng mảnh kiến trúc trước khi thiết kế mới** (theo user: không lắp ghép, phải hiểu thiết kế gốc).
  - Báo cáo [research/ARCH_ANALYSIS.md](research/ARCH_ANALYSIS.md) (F37) phân tích từng mảnh: thiết kế gốc, số đo trên v3 đã train, kết luận giữ / bỏ.
    Các mảnh: HGNetv2, LSK, hybrid encoder, head v3, YOLO11 / 26.
  - Phát hiện chính:
    - Đường P5 của backbone v3 chết ở cả 3 checkpoint (khoảng 8% FLOPs vô ích).
    - Lỗi chính là phân loại vật tí hon dựa vào ngữ cảnh, không phải định vị.
  - Bản nháp OpenOBB6 (refiner ngữ cảnh + adapter nét) đã gỡ khỏi repo theo lời user, chưa dùng.

- 2026-10-09 (9): **Rà soát và sửa cho infer + đọc data khớp Ultralytics.**
  - Góc `xywhr` nằm trong [0, π/2), theo quy ước `regularize_rboxes`; trước đây là [0, π) với w ≥ h. Tensor nằm trên device của model.
  - Ảnh đọc qua `openobb/data/imgio.py` (dùng `IMREAD_COLOR`, an toàn với Unicode) ở mọi chỗ: train, val, predict. Trước đây predict dùng `IMREAD_UNCHANGED`, bỏ qua xoay EXIF.
  - `predict`:
    - Tên tham số sai thì báo lỗi; tham số video thì cảnh báo.
    - Thêm `save_crop` và `show_boxes`; nhận `imgsz` dạng list.
    - Lưu ảnh kèm tên gốc; log `Speed:`.
    - Đọc lần lượt từng ảnh; nhận nguồn là file `.txt`.
  - CLI `openobb predict` đi qua `OpenOBB.predict` (mặc định `save=True`). Tham số cũ (`save_dir`, `vis`...) vẫn chạy đường cũ.
  - Dataset: thư mục ảnh quét cả thư mục con, `train:`/`val:` nhận file `.txt` hoặc list, nhãn tìm theo từng ảnh (`/images/` → `/labels/`), tên trùng không đè nhau. `PREP_VERSION` = 3, nên cache cũ sẽ được dựng lại.
  - `val`: P/R tính tại conf có F1 trung bình cao nhất (như YOLO), thêm `map75`, `.p`, `.r`, `.ap50`, `.ap`, `fitness`. Bảng P/R lúc train đổi theo.
  - Thêm `tests/test_ultralytics_compat.py`.

- 2026-10-09 (8): **Phiên bản.**
  - Một nguồn duy nhất là `openobb/__init__.py` (`__version__`); `pyproject.toml` đọc động từ đó.
  - `tools/push_all.py` tự tăng số patch khi `openobb/`, `pyproject.toml` hoặc `requirements.txt` đổi so với commit đặt version gần nhất.
    `--minor` để tăng số minor.
  - Gắn tag `vX.Y.Z` trên cả GitHub lẫn GitLab. Bản đổi tên OpenOBB là **0.2.0**.

- 2026-10-09 (7): **Đổi toàn bộ tên kiến trúc VRDet → OpenOBB** (theo user; bỏ hẳn tên cũ, không alias).
  - Tên mới:
    | Loại | Tên cũ → mới |
    |---|---|
    | Model | `model=vrdetNx` → `openobbNx` (`openobb1s..x`, `openobb2n..x`, `openobb3s..x`, `openobb4x`, `openobb5x`). Bỏ các tên tắt `s/m/l/x`. Mặc định `openobb1s` |
    | Class | `VRDet`, `VRDet2..5` (+Loss) → `OpenOBB1`, `OpenOBB2..5` (+Loss) |
    | File model | `openobb/models/vrdet*.py` → `openobb/models/openobb*.py` |
    | API deploy | `from openobb import OpenOBB` |
    | Tài liệu | `docs/ARCHITECTURE_CARD.md` → `OPENOBB1.md`; `VRDETn.md` → `OPENOBBn.md` |
    | Notebook | `colab/OpenOBB_*.ipynb` |
    | Test | `tests/test_openobbN.py` |
    | Thư mục run, cache | `openobb_args.json`, `~/.cache/openobb` |
    | Biến môi trường | `OPENOBB_SLOW` / `OPENOBB_NET` / `OPENOBB_CACHE` / `OPENOBB_DEBUG` |
  - Nhãn nội bộ `--arch v1..v5` giữ nguyên, nên checkpoint `.pt` cũ vẫn nạp được. Run cũ (`vrdet_args.json`) không resume được nữa.
  - Lịch sử (sổ cái, nhật ký ở trên, tên run nghiên cứu như `e1-vrdet-s-dota-24e`) giữ nguyên chữ VRDet. Đó là hồ sơ những gì đã xảy ra.
  - README của GitLab giờ lấy từ `tools/gitlab_README.md` (do `push_all.py` quản lý).

- 2026-10-09 (6): **Đổi tên package/CLI thành `openobb`**; kiến trúc vẫn tên VRDet (`vrdet5x`, class `VRDet`).
  - **Hai repo:**
    | Repo | Vai trò | Nội dung |
    |---|---|---|
    | GitHub `Nguyenchitrai62/OpenOBB` (đổi tên từ `vrdet`, URL cũ tự chuyển hướng) | Repo nghiên cứu, nơi phát triển | Đầy đủ: research, tools, colab, kaggle |
    | GitLab công ty `git.anybim.vn/CxDP/service/hicasai/openanything/openobb` | Repo sản phẩm, lịch sử mới | Chỉ phần sạch: `openobb/`, tests (bỏ 4 test cần script nghiên cứu), docs sản phẩm, app, LICENSE, notices, README riêng |
  - Chính sách của user: mỗi repo `open*` (sau này opencls, openocr, openpdf...) tự đứng riêng, không phụ thuộc phần cấm thương mại.
  - **Đẩy code: `python tools/push_all.py`.** Lệnh này đẩy GitHub (nhánh local `main`), rồi đồng bộ phần sạch sang GitLab:
    - dùng danh sách file cho phép + luật đổi chữ trong script;
    - chạy test trên bản GitLab, chỉ đẩy khi pass;
    - không force-push.

    Dùng `--dry-run` để xem trước. Colab cài từ GitHub, vì cài từ GitLab trên Colab rất khó.
  - Bài học 10-09: dòng `models/` trong `.gitignore` của GitLab từng nuốt mất `openobb/models/`.
    Giờ các luật ignore đã neo gốc (`/models/`), và script từ chối đẩy nếu thiếu file package.

- 2026-10-09 (5):
  - API deploy giống Ultralytics: `from openobb import OpenOBB` → `predict` → `Results.obb.*`, `plot`, `save_txt`, `summary`, `val().box.map`.
    Code ở `openobb/model.py`, `openobb/results.py`; tài liệu [docs/DEPLOY.md](docs/DEPLOY.md).
  - `rotate_p` giờ áp dụng cả lên ảnh mosaic. 94 test pass.

- 2026-10-09 (4):
  - VRDet5 (`openobb/models/vrdet5.py`, `vrdet5_loss.py`, `--arch v5`, `--geo-cls`, `--relate`, `--rel-k`).
  - v3 nhận `--backbone`. Confusion matrix trong `openobb/eval/ultra.py`. 90 test pass.

- 2026-10-09 (3):
  - Thước đo Ultralytics (`openobb/eval/ultra.py`).
  - VRDet4-x (`openobb/models/vit.py`, `DenseOriented`, `--arch v4`, `--backbone`, `--dense-v3`, `--primary`). 81 test pass.

- 2026-10-09 (2): VRDet3.
  - Viết `openobb/models/vrdet3.py`, `vrdet3_loss.py`. CLI nhận `vrdet3{s,m,l,x}`. `assign` dùng chung có dải đa tầng (v2 tắt).
  - Thêm `tests/test_openobb3.py`; 73 test pass.

- 2026-10-09: VRDet2.
  - Viết `openobb/models/vrdet2.py`, `vrdet2_loss.py`. CLI nhận `vrdet1*` / `vrdet2*`. Predict/val nạp được v2.
  - Test `tests/test_openobb2.py`; 65 test pass.
  - Overfit CPU tìm ra lỗi gán nhãn cho line dài, đã sửa (LEDGER F24).

- 2026-10-08 (chiều, chốt):
  - Đợt raster-CAD trên FloorPlanCAD (S, 24 epoch, so c4 = 76.83 / 66.57):
    - c5 P2 stride-4: bác bỏ (74.04 / 63.32).
    - **c6 LSK: 78.16 / 68.81, xác nhận**, thành kiến trúc chốt.
  - Phân rã khoảng cách với YOLO26x (F18): recall ngang; thua ở độ khít box object mảnh và xếp hạng symbol nhỏ; bản S nhỏ hơn 4.6 lần.
  - Đóng gói thư viện:
    - `pyproject.toml`, `openobb/cli.py`, `openobb/data/prepare.py` + `split.py`.
    - `best.pt` cùng fine-tune giữ class theo tên.
    - `colab/OpenOBB_train.ipynb` thay `colab/finetune_vrdet.ipynb`.
  - Push GitHub `Nguyenchitrai62/OpenOBB` (public theo lời user).
  - Tiêu khoảng 30 CU cho c4/c5/c6; còn khoảng 71.7 CU.

- 2026-10-08 (đêm):
  - Mốc YOLO26x DOTA val (24 epoch) **78.42**. VRDet-X đạt **75.61** (−2.81; mAP50:95 −0.79; 64% FPS).
  - Context thumbnail bị bác bỏ (−3.88). Head dense chưa có lợi, bản thân nhánh dense hỏng.
  - Nhánh vector H7 trên FloorPlanCAD: +0.84 mAP50, +2.51 mAP50:95.
  - 600 query lúc suy luận cho +0.7 (SV +6.3).
  - Phát hiện yếu tố gây nhiễu mosaic (YOLO có, VRDet không), đã soạn E10.
  - Hạ tầng: giới hạn 3 session; download resume theo chunk; quy tắc CU 170/130; `tools/summarize.py`; `tools/prepare_dataset.py` (finetune data riêng).
  - Tiêu khoảng 275 CU từ đầu, còn khoảng 225.

- 2026-10-07 (4):
  - Nghiên cứu web bằng 6 sub-agent. **Mốc mới: RiO-DETR** (ECCV 2026, Apache) đạt SS s 80.3 / x 81.8, nhỉnh hơn YOLO26.
  - Hai model mạnh ở chỗ khác nhau: DETR thắng class lớn cần ngữ cảnh, YOLO thắng vật nhỏ. Lấy max theo class được 82.3, nên chốt hướng **dense–sparse hybrid**.
  - Đổi license: DEIMv2 đã thành NC; code O2-RTDETR (ai4rs) chép từ RHINO (NC), nên chỉ lấy ý tưởng.
  - Viết evaluator, script data, VRDet v0, head dense. Launch baseline YOLO26s.

- 2026-10-07 (3): Xây khung tự động không cần người (cx.py, job.py, AUTONOMY.md, skill vrdet-research). Phát hiện: upload CLI rớt với file >~20–80 MB → chia chunk 16 MB + sha256; exec có thể treo khi 2 tiến trình poll cùng lúc → thêm lock + chỉ coi VM chết khi session biến mất hoặc 3 lần lỗi liên tiếp. G4 đo được 195 TFLOPS bf16. 2 lần smoke tiêu ~4 CU.

- 2026-10-07 (2): User chốt: Pro+ 500 CU, mục tiêu = kiến trúc OBB tổng quát benchmark trên DOTA như YOLO, accuracy trước. Viết RESEARCH_PLAN + COLAB. Colab CLI chạy được trên Windows sau 2 bản vá. Mốc phải vượt: YOLO26-OBB (x: 81.7 mAP50 DOTA test); DETR-OBB Apache tốt nhất đang biết: O2-DEIM-R50 80.15.

- 2026-10-07: Bắt đầu vòng nghiên cứu mới (VRDet). Kết luận vòng cũ thất bại do tự chế + from scratch. Đề xuất hướng hybrid vector+raster trên nền DETR-OBB Apache-2.0.
