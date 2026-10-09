# Colab CLI trên Windows: hướng dẫn cho agent

Tài khoản: Colab **Pro+**. Ngày 2026-10-07 balance ~503 compute units (CU). Xem số dư hiện tại bằng `colab usage`.

## 1. Cài đặt (đã làm trên máy này)

```bash
uv tool install google-colab-cli      # 0.7.4
```

Bản gốc chỉ hỗ trợ Linux/macOS. Trên máy này đã có **2 bản vá local** (bản gốc giữ ở file `*.orig` cạnh đó), trong `%APPDATA%\uv\tools\google-colab-cli\Lib\site-packages\colab_cli\`:

1. `console.py`: bọc `import termios` / `import tty` trong try/except. Mọi lệnh chạy được, trừ `colab console`.
2. `commands/automation.py`: `drivemount` không đọc `/dev/tty` nữa mà **poll tối đa 5 phút** chờ user cấp quyền Drive trên trình duyệt.

Sau `uv tool upgrade` thì phải vá lại: so với file `.orig`, hoặc làm lại theo mô tả trên.

## 2. Biến môi trường bắt buộc khi gọi từ Git Bash

```bash
export PYTHONIOENCODING=utf-8   # CLI in ký tự Unicode
export MSYS_NO_PATHCONV=1       # nếu thiếu, Git Bash đổi /content/... thành C:/Program Files/Git/content/...
```

Từ PowerShell thì chỉ cần `$env:PYTHONIOENCODING='utf-8'`.

## 3. Đăng nhập (oauth2, user tự đăng nhập Google)

Luồng gốc của CLI cần dán mã vào stdin, nên agent dùng script tách 2 bước [tools/colab_login.py](../tools/colab_login.py):

```bash
PY="$APPDATA/uv/tools/google-colab-cli/Scripts/python.exe"
"$PY" tools/colab_login.py url        # in ra URL đăng nhập → mở cho user (browser pane)
```

User đăng nhập xong thì Google hiện mã. **Agent không tự đọc/nhập mã này**. User bấm Copy rồi tự chạy trong PowerShell:

```powershell
Get-Clipboard | & "$env:APPDATA\uv\tools\google-colab-cli\Scripts\python.exe" F:\Source_code\NEW_architecture\tools\colab_login.py exchange
```

Token lưu ở `~/.config/colab-cli/token.json` (có refresh token, tự gia hạn). Kiểm tra bằng `colab whoami`. Tài khoản đang dùng: `nam@pose3d.ai`.

## 4. Lệnh hay dùng

```bash
colab new -s <name> --gpu T4|L4|G4|A100|H100 [--high-mem]   # luôn đặt -s
colab exec -s <name> -f script.py            # gửi file local lên chạy, không cần upload
echo "print(1)" | colab exec -s <name>       # chạy đoạn code ngắn
colab upload -s <name> LOCAL /content/x      # dùng path tuyệt đối /content/...
colab download -s <name> /content/x LOCAL
colab install -s <name> -r requirements.txt
colab drivemount -s <name> /content/drive    # lần đầu cần user bấm Allow trên trình duyệt
colab sessions | colab status -s <name> | colab usage
colab stop -s <name>                         # LUÔN stop khi xong để khỏi tốn CU
colab run --gpu A100 script.py args          # VM tạm: new + exec + stop
```

Ghi chú:
- Kernel giữ state giữa các lần `exec` trong cùng session.
- Đừng chạy `repl`/`console` tương tác từ agent. Nếu cần thì pipe stdin vào.
- Job dài (train nhiều giờ): chạy dạng `subprocess.Popen(..., stdout=log)` + `nohup` trong VM để `exec` trả về ngay. Sau đó poll log bằng `colab exec` định kỳ, đừng giữ websocket hàng giờ.

## 5. Quy ước cho nghiên cứu

- Tên session: `<exp-id>-<gpu>`, ví dụ `e1-dior-a100`.
- **Mọi checkpoint/log ghi vào Drive**: `/content/drive/MyDrive/openobb/runs/<exp-id>/`. Train phải resume được từ `last.pt` vì VM có thể chết bất cứ lúc nào.
- Dataset: tải 1 lần, nén, để trên Drive tại `/content/drive/MyDrive/openobb/datasets/`. Mỗi VM mới copy về `/content/datasets` (đĩa local ~190 GB, nhanh hơn đọc trực tiếp từ Drive).
- Ghi CU tiêu thụ của mỗi thí nghiệm vào `research/decisions.log`.
- Môi trường VM đo ngày 2026-10-07 (T4): Python 3.13, torch 2.11.0+cu130, driver 580.

## 6. Chi phí

Đo thực tế bằng `colab usage` (CU/giờ của các máy đang chạy). Ngày 2026-10-07, khi đang có 1 A100 chạy (session của user, không phải của agent), usage rate là 5.30 CU/h. Cập nhật bảng dưới khi đo được thêm:

| GPU | CU/giờ (đo 2026-10-07) | Dùng cho |
|---|---|---|
| T4 | ~1.1 (6.37 − 5.30) | smoke test rẻ |
| **G4** = RTX PRO 6000 Blackwell, 94 GB, ~195 TFLOPS bf16 (đo) | **~8.9** (14.20 − 5.30) | **GPU mặc định cho nghiên cứu** (user chọn) |
| A100 | ~5.3 | dự phòng khi G4 hết chỗ |
| H100 | ? | chưa đo |

500 CU ≈ **~55 giờ G4**.

**Giới hạn đồng thời (đo 2026-10-07):**
- Colab chỉ cấp khoảng **3 G4 cùng lúc**. VM thứ tư báo "Allocation refused (precondition failed)".
- Lúc đó A100 vẫn cấp được (đã chạy 2 A100 song song với 3 G4).
- A100 rẻ hơn mỗi giờ (~5.3 so với ~8.9 CU/h). Bộ nhớ 40 GB đủ cho OpenOBB1-S batch 16 ở 1024 (khoảng 23 GB).
- `tools/job.py queue <id>` đưa job vào hàng đợi; watcher tự launch khi có chỗ. Hãy ước lượng CU trước mỗi job (giờ × 8.9) và ghi vào `decisions.log`.
