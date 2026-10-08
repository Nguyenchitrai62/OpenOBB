@echo off
REM Mo app test model VRDet (keo tha anh). Dat best.pt vao thu muc models\ truoc.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Chua co moi truong .venv - xem docs\FINETUNE.md muc 4
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py
pause
