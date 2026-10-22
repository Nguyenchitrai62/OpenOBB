# Kiểm tra pháp lý / bản quyền repo OpenOBB (2026-10-22)

> Mục đích gốc của repo: thay YOLO (Ultralytics, AGPL-3.0) bằng một detector OBB thương mại được, thuộc sở hữu của chủ repo.
> Tài liệu này ghi lại kết quả kiểm tra bằng công cụ trên toàn bộ code, weights, dataset và cách đóng gói, kèm việc cần làm.
> Đây là phân tích kỹ thuật, **không phải tư vấn pháp lý**; với hợp đồng thương mại lớn nên có luật sư rà lại.

## 1. Kết luận ngắn

| Câu hỏi | Trả lời |
|---|---|
| Code sản phẩm (`openobb/`, `openobb_onnx/`, `app/`) có import Ultralytics / thư viện AGPL hay NC không? | **Không.** Chỉ torch, numpy, opencv, shapely, scipy, PIL, yaml, onnxruntime, streamlit (BSD / Apache / MIT / HPND). Test `tests/test_license_guard.py` và `push_all.py` chặn import cấm. |
| Có đoạn code nào **chép** từ Ultralytics không? | **Có, nhưng rất ít và nhỏ**: 3 hàm ở mức 40–66 % trùng token (`_smooth`, `_ap` trong `openobb/eval/ultra.py`; `_hsv` trong `openobb/data/dota.py`), cộng 1 khối trong `hybrid_encoder.py` mà **D-FINE đã chép từ YOLOv9 / Ultralytics** trước khi mình mượn D-FINE. Tổng cộng khoảng 20 dòng. Sửa được trong một buổi. |
| "Khung training giống YOLO" có vi phạm bản quyền không? | **Không**, miễn là không chép *văn bản code*. Hyper-parameter, lịch LR, cách đặt tên tham số (`imgsz=`, `conf=`), cấu trúc `runs/obb/train`, cách tính mAP, giao diện `Results` là **ý tưởng / phương pháp / giao diện**, không được bảo hộ bản quyền (Luật SHTT VN Điều 15; án lệ Google v. Oracle ở Mỹ). Kết quả đo: `train.py`, `engine.py`, `cli.py`, `model.py`, `predict.py` không có hàm nào trùng Ultralytics trên 27 %, và các hàm trùng đó đều là mẫu chung (EMA, Conv-BN-act). |
| Giấu code training, chỉ giao model + infer: "đã ổn chưa"? | **Ổn về mặt sản phẩm giao khách, nhưng lý do không phải "giấu".** AGPL chỉ áp dụng khi *phân phối* hoặc *cho người dùng tương tác qua mạng* với **code dẫn xuất của Ultralytics**. Repo không có code dẫn xuất đó trong trainer, nên trainer dùng nội bộ hay công khai đều không vi phạm. Ngược lại, 2 trong 4 chỗ cần sửa lại nằm trong **đường val / infer** (`eval/ultra.py`), là phần bạn tưởng sạch. |
| Weights v23 có sạch không? | **Có.** Lineage: D-FINE COCO (Apache-2.0) → OpenOBB12 train trên Wall_Color (data riêng) → v17 → v18 (neck tự viết, chưng cất từ chính v12 / v17) → v19 → v20 → v21 → v23. Không có thầy YOLO, không có DOTA / FloorPlanCAD trong lineage. Không có file `.pt` nào được track trong git. |
| Giấy tờ đi kèm wheel giao khách đủ chưa? | **Chưa.** Wheel ONNX và wheel đầy đủ chỉ chứa `LICENSE`, thiếu `THIRD_PARTY_NOTICES.md` + văn bản Apache-2.0 mà Apache §4 yêu cầu khi phân phối (code HGNetv2 port + weights khởi tạo từ D-FINE). |

## 2. Cách kiểm tra

1. Đọc `LICENSE`, `THIRD_PARTY_NOTICES.md`, `LICENSES/`, `pyproject.toml`, `requirements.txt`, `tests/test_license_guard.py`, `tools/push_all.py`, `tools/build_onnx_wheel.py`.
2. Liệt kê mọi `import` trong `openobb/`, `openobb_onnx/`, `app/` (AST).
3. **So khớp token** từng hàm của ba package trên với mã nguồn Ultralytics 8.4.174 (bản cài local để đo), D-FINE, DEIM, RT-DETR (clone về scratchpad). Thước: tỉ lệ 6-gram token trùng, hai chế độ *exact* (giữ tên biến) và *shape* (chuẩn hoá tên biến, bắt bản "đổi tên"). Script: `research/analysis/license_simcheck.py`.
   ```bash
   py -3.12 -I research/analysis/license_simcheck.py openobb ultralytics=<site-packages>/ultralytics dfine=<clone> --thr=0.25
   ```
4. Grep chú thích nguồn gốc, URL tải weights / data, lineage checkpoint trong `research/LEDGER.md`, nội dung wheel đã build.

## 3. Phát hiện, xếp theo mức rủi ro

### 3a. Cần sửa (chép văn bản code từ Ultralytics, AGPL)

| # | File | Trùng (exact / shape) | Nguồn | Có trong bản giao khách? | Sửa |
|---|---|---|---|---|---|
| 1 | `openobb/eval/ultra.py` `_smooth` (4 dòng) | 0.66 / 0.88 | `ultralytics/utils/metrics.py::smooth` (gốc YOLOv5, AGPL/GPL) | GitLab + wheel đầy đủ (val); **không** trong wheel ONNX | Viết lại: ví dụ `np.convolve` với padding `np.pad(mode="edge")`, hoặc `scipy.ndimage.uniform_filter1d(mode="nearest")`. |
| 2 | `openobb/eval/ultra.py` `_ap` | 0.44 / 0.60 | `metrics.py::compute_ap` (sentinel + envelope từ py-faster-rcnn MIT, dòng nội suy 101 điểm là của YOLOv5) | như trên | Viết lại theo định nghĩa COCO 101 điểm: với mỗi `r` trong `linspace(0,1,101)` lấy `max(prec[rec >= r])`, rồi `mean`. Cùng số, khác biểu đạt. |
| 3 | `openobb/data/dota.py` `_hsv` | 0.42 / 0.49 | `data/augment.py::RandomHSV` (mẫu `augment_hsv` YOLOv5: LUT 3 kênh) | Chỉ train (GitLab + wheel đầy đủ có `train.py`) | Viết lại: cộng hue / nhân sat, val trực tiếp trên mảng HSV float rồi `clip` + `astype`, không qua `cv2.LUT`. |
| 4 | `openobb/models/hybrid_encoder.py` khối "csp-elan" (`forward`, `forward_chunk`, 3 dòng mỗi hàm) + `VGGBlock.get_equivalent_kernel_bias` | 0.90 / 0.91 và 0.78 / 0.87 | Ultralytics `RepNCSPELAN4` (YOLOv9, GPL-3.0) và `RepConv` (gốc RepVGG, MIT). **D-FINE upstream cũng trùng đúng mức này** (`refs/dfine/src/zoo/dfine/hybrid_encoder.py`), tức D-FINE chép từ YOLOv9 / Ultralytics rồi gắn nhãn Apache; họ không có quyền đổi license đoạn đó. | Code có trong package; **v23 không dùng** (OpenOBB18+ `del self.encoder`); wheel ONNX không chứa code model | Viết lại 2 hàm forward bằng cách khác (ví dụ `torch.split` gán tên rõ, không `list.extend` generator), và ghi vào NOTICES rằng khối RepVGG theo RepVGG (MIT). Phần `RepConv` fuse thực ra gốc MIT (DingXiaoH/RepVGG) nên chỉ cần ghi nguồn. |

Mức độ: mỗi chỗ là vài dòng thuật toán quen thuộc, khó kiện riêng lẻ (de minimis / scènes à faire), nhưng vì repo công khai trên GitHub và mục tiêu là "không dính Ultralytics", nên dọn sạch cho khỏi tranh cãi. Sau khi sửa, chạy lại script ở mục 2 với `--thr=0.25` phải không còn dòng `ultralytics` nào ngoài tên trùng tình cờ.

### 3b. Cần bổ sung giấy tờ (Apache-2.0 §4, không phải vi phạm bản quyền nhưng là vi phạm điều kiện license)

| # | Việc | Ở đâu |
|---|---|---|
| 5 | Wheel ONNX (`tools/build_onnx_wheel.py`) chỉ copy `LICENSE`. Thêm `THIRD_PARTY_NOTICES.md` + `LICENSES/Apache-2.0-D-FINE.txt` vào wheel (ví dụ `<name>/licenses/`). Lý do: file `.onnx` chứa weights HGNetv2 khởi tạo từ D-FINE COCO; Apache yêu cầu kèm license + notice khi phân phối tác phẩm dẫn xuất. | `tools/build_onnx_wheel.py` |
| 6 | Wheel đầy đủ (`pyproject.toml`) không khai `package-data` / `MANIFEST.in` cho `THIRD_PARTY_NOTICES.md`, `LICENSES/`. Thêm vào `[tool.setuptools]` hoặc `license-files = ["LICENSE", "THIRD_PARTY_NOTICES.md", "LICENSES/*"]`. | `pyproject.toml` |
| 7 | `THIRD_PARTY_NOTICES.md` ghi "code is OpenOBB's own" cho `openobb/eval/ultra.py` và `results.py`; sau khi sửa mục 3a thì đúng; hiện chưa đúng. Thêm dòng cho RepVGG (MIT) và DINOv2 (chỉ v4, không giao). | `THIRD_PARTY_NOTICES.md` |
| 8 | Header của `hybrid_encoder.py` chỉ ghi DEIM / D-FINE; thêm một dòng "contains a CSP-ELAN block rewritten by OpenOBB; RepVGG re-parameterisation after DingXiaoH/RepVGG (MIT)". | `openobb/models/hybrid_encoder.py` |

### 3c. Thấp, chỉ ghi nhận

| # | Điểm | Nhận định |
|---|---|---|
| 9 | `openobb/results.py` và `openobb_onnx/results.py`: `verbose`, `save` trùng 0.43–0.49 exact với Ultralytics | Hàm 3–4 dòng, trùng vì **cố ý giữ cùng chuỗi đầu ra** (`"(no detections), "`, `results_<tên ảnh>`) để tương thích. Giao diện / chuỗi ngắn không được bảo hộ. Giữ. |
| 10 | `engine.py` `ModelEMA.update` 0.27 exact | Công thức ramp `decay·(1−e^{−t/τ})` là của YOLOv5, nhưng là công thức toán; code viết khác (`_foreach`). Giữ. |
| 11 | `models/dense_head.py` `ConvBN.__init__` 0.62 exact | 3 dòng Conv-BN-act, mẫu chung của mọi repo. Giữ. |
| 12 | `ops/obb_torch.py::probiou` 0.17, `norm_le90` 0.13 | Thấp; ProbIoU theo paper (Murrugarra-Llerena 2022). Giữ. |
| 13 | `openobb/eval/dota.py::voc_ap` 0.28 với `compute_ap` | Cả hai cùng gốc py-faster-rcnn `voc_ap` (MIT). Giữ. |
| 14 | `openobb/data/download.py` tải `DOTAv1.zip` từ `github.com/ultralytics/assets` | Chỉ là mirror **dữ liệu**, không phải code. DOTA là license học thuật → chỉ dùng cho `pretrain=dota` nghiên cứu, không cho weights thương mại. Đã ghi trong notices. |
| 15 | `colab/baselines/yolo_obb.py`, `colab/speed_gpu.py` import `ultralytics` | Script baseline trong repo nghiên cứu, không vào GitLab / wheel (`push_all.py` FORBIDDEN). Chạy Ultralytics để đo là quyền sử dụng hợp lệ dưới AGPL. Giữ, nhưng **không bao giờ** copy sang repo sản phẩm. |
| 16 | D-FINE / DEIM / RT-DETR (Apache-2.0): 150 hàm trùng trong `common.py`, `deim_utils.py`, `hgnetv2.py`, `hybrid_encoder.py`, `obb_decoder.py`, `obb_criterion.py`, `obb_utils.py` | Đúng như `THIRD_PARTY_NOTICES.md` khai, không có file nào mượn mà chưa khai. Apache-2.0 cho phép thương mại, đóng nguồn; chỉ cần giữ notice + ghi "đã sửa" (đã có ở header mỗi file). |
| 17 | Weights khởi tạo: `dfine_{s,m,l,x}_coco.pth`, `PPHGNetV2_B*_stage1.pth` (ImageNet) từ release của tác giả D-FINE; DINOv2 (Apache, chỉ v4) | Apache-2.0. Đã tránh bản Objects365 (điều khoản riêng). COCO: ảnh Flickr, nhãn CC BY 4.0; weights pretrain COCO được ngành dùng thương mại rộng rãi (chính YOLO cũng vậy). |
| 18 | `models/best_v1.pt`, `models/vrdet_s_lsk_floorplancad.pt`, `runs/*/last.pt` (train trên FloorPlanCAD CC BY-NC, DOTA) | Gitignored, không track, không push. Không được giao khách hay làm init cho weights thương mại. |
| 19 | Nhãn hiệu | Tên sản phẩm `HicasOBB` / `OpenOBB` không chứa "YOLO" / "Ultralytics". Nhắc "YOLO" trong README để so sánh là dùng nhãn hiệu kiểu chỉ danh (nominative), hợp lệ; tránh dùng logo Ultralytics. |
| 20 | Bằng sáng chế | Apache-2.0 kèm grant bằng sáng chế của tác giả D-FINE / RT-DETR cho phần code họ đóng góp. Không rõ Baidu có đăng ký sáng chế riêng cho RT-DETR không; rủi ro thấp cho phần v23 vì encoder lai đã bị bỏ, chỉ còn thân HGNetv2 (PaddlePaddle, Apache). Không có sáng chế công khai nào của Ultralytics về quy trình train. |
| 21 | PyMuPDF (AGPL) | Không có trong repo này (`fitz` không được import). AI_Takeoff dùng PyMuPDF: đó là rủi ro AGPL **riêng của AI_Takeoff** (dịch vụ qua mạng = AGPL §13), không liên quan OpenOBB. Nếu muốn sạch toàn chuỗi: pypdfium2 (Apache / BSD) hoặc mua license Artifex. |
| 22 | AI_Takeoff hiện gọi YOLO server ngoài | Nếu server đó chạy Ultralytics và khách tương tác qua mạng, AGPL §13 áp lên **server đó**. Thay bằng OpenOBB ONNX đúng là cách thoát. |

## 4. Nguyên tắc để tự quyết các lần sau

1. **Bản quyền bảo hộ biểu đạt, không bảo hộ ý tưởng.** Chép / dịch sát từng dòng code = vi phạm. Đọc hiểu rồi tự viết lại theo mô tả hành vi (như đã làm với `results.py`, `model.py`, `prepare.py`) = không vi phạm. Tên tham số, giá trị mặc định, thứ tự in log, cấu trúc thư mục không được bảo hộ.
2. **AGPL chỉ kích hoạt khi "convey" hoặc "network interaction" với tác phẩm dẫn xuất.** Dùng Ultralytics nội bộ để đo baseline: hợp lệ. Dùng code Ultralytics đã sửa làm dịch vụ cho khách qua mạng: phải mở mã. Repo hiện không có code dẫn xuất Ultralytics trong sản phẩm, nên câu hỏi "giấu trainer" không phát sinh.
3. **Weights.** Weights train bằng code của mình, khởi tạo từ weights Apache, trên data của mình = của mình. Ultralytics tuyên bố weights train bằng code của họ là tác phẩm dẫn xuất (quan điểm gây tranh cãi, chưa có án lệ); OpenOBB không chạm vào trường hợp này vì không dùng code lẫn weights của họ, kể cả làm thầy chưng cất (đã kiểm tra `LEDGER`, `docs/OPENOBB18–20.md`).
4. **Apache-2.0 có điều kiện**: giữ LICENSE + NOTICE, ghi file đã sửa. Vi phạm điều kiện này là lỗi hình thức, sửa bằng giấy tờ, nhưng phải làm trước khi giao hàng (mục 3b).
5. **Data NC / học thuật** (FloorPlanCAD, DOTA): chỉ nghiên cứu, benchmark. Weights giao khách phải có lineage không đi qua chúng (v23 đạt).

## 5. Sở hữu chính code của mình (để giấy phép proprietary có hiệu lực)

- `LICENSE` còn ghi "the repository owner": điền đúng pháp nhân (cá nhân hay công ty). Nếu code viết trong thời gian làm việc theo nhiệm vụ công ty giao, theo Luật SHTT VN Điều 39 công ty là chủ sở hữu quyền tài sản mặc định; nên có văn bản chuyển giao / xác nhận giữa cá nhân (tài khoản GitHub `Nguyenchitrai62`) và công ty (GitLab `anybim`) để tránh tranh chấp sau này.
- Phần lớn code do agent AI sinh ra theo chỉ đạo của bạn. Luật VN yêu cầu tác giả là cá nhân; nhiều nước (Mỹ) không cấp bản quyền cho phần thuần AI sinh. Điều này **không tạo rủi ro bị kiện**, nhưng có thể làm giấy phép proprietary khó cưỡng chế với phần đó. Cách giảm: giữ lịch sử commit, thẻ kiến trúc `docs/OPENOBBn.md` (chứng minh thiết kế, lựa chọn, chỉnh sửa của con người), và bảo vệ bằng hợp đồng / bí mật kinh doanh (không công khai GitLab, NDA với khách) thay vì chỉ dựa vào bản quyền.
- Repo GitHub công khai với license proprietary: hợp lệ, nhưng mọi người xem được. Nếu muốn giữ bí mật kinh doanh thì chuyển private; Colab vẫn cài được bằng token.

## 6. Việc cần làm, theo thứ tự

1. Viết lại 4 chỗ ở mục 3a (khoảng 20 dòng), chạy lại `tests/` và `license_simcheck.py`.
2. Đóng gói notices vào cả hai wheel (mục 5, 6), cập nhật `THIRD_PARTY_NOTICES.md` (mục 7, 8).
3. Điền chủ sở hữu vào `LICENSE`; làm giấy chuyển giao cá nhân → công ty nếu cần.
4. Quy trình giữ sạch về sau: mỗi lần thêm code "giống YOLO", chạy `license_simcheck.py` với `--thr=0.25`; mọi dòng `ultralytics` xuất hiện phải viết lại trước khi `push_all.py`. Có thể thêm bước này vào `push_all.py` (chạy script trên bản export, fail nếu có hàm trùng trên 0.4 exact).
5. Tuỳ chọn: giữ `train.py` nội bộ như bạn định (bỏ `openobb/train.py`, `engine.py`, `data/dota.py`, `auto_pretrain.py`, `pretrain.py` khỏi `ALLOW` của `push_all.py`). Về pháp lý không bắt buộc; về kinh doanh thì bớt lộ bí quyết. Nếu làm, `openobb val` vẫn cần `eval/ultra.py` nên vẫn phải sửa mục 1 và 2 ở 3a.

## 7. Đã sửa (cùng ngày, theo yêu cầu user)

| Việc | File | Kiểm chứng |
|---|---|---|
| Viết lại `_ap`, `_smooth` | `openobb/eval/ultra.py` | giống bản cũ tới 1e-12 trên 20 đường PR / F1 ngẫu nhiên (snapshot trước khi sửa); `tests/test_license_rewrites.py` so với định nghĩa viết bằng vòng lặp |
| Viết lại `_hsv` | `openobb/data/dota.py` | giống bản cũ **từng bit** trên 5 ảnh / 5 seed (cùng cách dùng RNG, cùng phép toán float64 rồi cắt uint8) |
| Viết lại khối CSP-ELAN (`RepNCSPELAN4` → `CSPELAN`, giữ alias tên cũ) và `get_equivalent_kernel_bias` | `openobb/models/hybrid_encoder.py` | đầu ra giống bản cũ tới 1e-5 trước và sau khi gộp RepVGG; tên tham số `cv1..cv4` giữ nguyên nên checkpoint D-FINE / OpenOBB1–17 nạp y nguyên |
| Viết lại thêm `forward` kiểu C2f của `CSP` / `CSPStage` / `RepCSP` (v2 / v6 / v9), phát hiện khi chạy lại script sau sửa | `openobb/models/openobb2.py`, `openobb6.py`, `openobb9.py` | test so với định nghĩa, `torch.equal` |
| Wheel đầy đủ và wheel ONNX kèm `THIRD_PARTY_NOTICES.md` + `LICENSES/*` trong `dist-info/licenses/` (PEP 639, `setuptools>=77`) | `pyproject.toml`, `tools/build_onnx_wheel.py` | `tests/test_onnx.py` build wheel |
| Notices: hàng RepVGG (MIT, ý tưởng), ghi rõ phần "viết từ hành vi Ultralytics" và đợt rà 2026-10-22 | `THIRD_PARTY_NOTICES.md`, header `hybrid_encoder.py` | |

Chạy lại `license_simcheck.py` sau khi sửa (`--thr=0.25`, so với Ultralytics): không còn hàm nào của 4 mục 3a. Các hàm còn trùng ≥ 0.25
đều đã xem và giữ, vì cùng tổ tiên Apache / MIT hoặc là mẫu 1–3 dòng: `hgnetv2.forward` (PaddleDetection), `obb_criterion._src_idx` và
`obb_decoder` (DETR / RT-DETR, Ultralytics cũng port từ đó), `_fuse_bn_tensor` (RepVGG MIT), `ConvBN.__init__`, `results.verbose / save`
(chuỗi tương thích), `engine.ModelEMA.update` (công thức), `eval/dota.voc_ap` (py-faster-rcnn MIT).

Không đổi: CLI, API, định dạng checkpoint, siêu tham số, số mAP của mọi mục F trong sổ cái.
