@echo off
REM Mo app test model OpenOBB (keo tha anh).
REM ===== Doi model o day (duong dan .pt). De trong thi dung MODEL_PATH trong app\streamlit_app.py / thu muc models\ =====
set "MODEL=F:\Downloads\floorplan_new.pt"
REM ==========================================================================================================
cd /d "%~dp0"
set "OPENOBB_MODEL=%MODEL%"
if not exist ".venv\Scripts\python.exe" (
  echo Chua co moi truong .venv - xem docs\FINETUNE.md muc 4
  pause
  exit /b 1
)
if defined MODEL if not exist "%MODEL%" echo Canh bao: khong thay model %MODEL%
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
pause
