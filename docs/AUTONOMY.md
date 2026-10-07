# Giao thức nghiên cứu tự động (không cần user online)

User chỉ thỉnh thoảng vào kiểm tra. Agent **tự quyết định và tự chạy** toàn bộ vòng nghiên cứu theo giao thức dưới đây. Không bước nào được phụ thuộc vào việc người bấm nút.

## 1. Vòng lặp mỗi lần agent "thức dậy"

Agent thức dậy vì một trong ba lý do: watcher kết thúc, task định kỳ chạy, hoặc user nhắn. Mỗi lần thức dậy:

1. Đọc `AGENTS.md` (mục 8–9), `research/HYPOTHESES.md`, rồi chạy `python tools/job.py status`.
2. Nếu có job vừa kết thúc:
   - đọc `runs/<id>/metrics.jsonl` và `train.log`,
   - ghi kết luận (xác nhận/bác bỏ giả thuyết, kèm số) vào `research/HYPOTHESES.md`,
   - ghi một dòng vào `research/decisions.log` (thời gian, job, kết quả, CU đã tiêu, quyết định tiếp).
3. Chọn thí nghiệm tiếp theo theo lộ trình trong `docs/RESEARCH_PLAN.md` mục 5:
   - viết giả thuyết + tiêu chí pass/fail **trước**,
   - tạo `research/jobs/<id>.json`, chạy `python tools/job.py launch <id>`.
4. Chạy watcher nền: `python tools/job.py watch` (Bash `run_in_background`). Watcher thoát khi một job kết thúc hoặc sau 8 giờ, và việc thoát đó đánh thức agent.
5. Cập nhật `AGENTS.md` mục 8–9, rồi commit git.

## 2. Khung kỹ thuật (đã kiểm chứng trên Colab thật)

- [tools/cx.py](../tools/cx.py): wrapper cho Colab CLI, có retry, upload chia chunk 16 MB (CLI rớt khi một file vượt khoảng 20–80 MB), verify sha256, đọc số dư CU.
- [tools/job.py](../tools/job.py): `launch / poll / watch / stop / status`.
  - Mỗi job chạy trên VM riêng. Code được ship dạng tar, các bước setup idempotent, lệnh train chạy nền (`nohup`) trên VM.
  - Mỗi lần poll: đồng bộ checkpoint/metrics/log về `runs/<id>/` tại local.
  - **VM chết** → tạo VM mới, push `last.pt` từ local lên, train tự resume. Tối đa `max_retries` lần.
  - Kết thúc (thành công, lỗi, hay quá `max_hours`) → đồng bộ lần cuối rồi `colab stop`.
- **Không dùng Google Drive**: mount Drive bắt user bấm Allow mỗi VM. Thay vào đó:
  - dataset public: VM tự tải từ internet (script trong `colab/data/`, idempotent),
  - data local: `cx.put` chia chunk,
  - checkpoint: đồng bộ về local (đo được ~12 MB/s, 200 MB mất 17 s).

## 3. Luật tự quyết (agent được phép)

- Chọn, sửa, huỷ thí nghiệm; viết/sửa code trong repo; tạo/tắt VM **do chính agent tạo** (session tên theo job id).
- Chọn GPU: **G4 (RTX PRO 6000 Blackwell 94 GB, user chọn) là mặc định** cho ablation và train. T4 chỉ dùng cho smoke test/debug rẻ. A100 là dự phòng khi G4 không cấp được.

## 4. Luật cứng (KHÔNG được vi phạm)

- **Ngân sách**: không launch nếu số dư < 40 CU (watcher tự stop mọi job khi chạm ngưỡng). Mỗi job phải có `max_hours`. Ghi CU tiêu thụ vào `decisions.log`.
- **Không đụng session lạ**: các dòng `[?]` trong `colab sessions` là của user. Không stop, không exec vào đó.
- Không bao giờ để VM của agent chạy không việc gì. Trước khi kết thúc lượt, kiểm tra `colab sessions`.
- Không claim kết quả khi chưa có số. Ghi trung thực cả thất bại.
- Không đăng nhập, nhập mật khẩu/mã OAuth thay user. Nếu token Colab hỏng (`colab whoami` lỗi): dừng, ghi rõ vào `AGENTS.md` mục 8 cách đăng nhập lại (docs/COLAB.md mục 3), chờ user.
- Không commit secret, dataset, weights.

## 5. Việc vẫn cần user (gom lại, user xem khi ghé qua)

Ghi vào `AGENTS.md` mục 8 dưới tiêu đề **"Cần user"**:
- đăng nhập lại Colab nếu token hết hạn,
- tạo tài khoản server đánh giá DOTA test (bước E4),
- nạp thêm compute units.
