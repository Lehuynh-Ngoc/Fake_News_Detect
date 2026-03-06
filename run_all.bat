@echo off
title Vietnamese Fake News Detection System (All-in-One)
echo ======================================================
echo   DANG KHOI CHAY HE THONG (MOI THU TRONG 1 CUA SO)
echo ======================================================
echo.

:: 1. Khoi chay Backend trong nen (background) cua cua so nay
echo [+] Dang khoi chay Backend (FastAPI)...
start /b python main.py

:: 2. Doi mot chut de Backend san sang, sau do mo trinh duyet
echo [+] Dang chuan bi mo trinh duyet...
timeout /t 5 /nobreak > nul
start "" "http://localhost:5173"

:: 3. Khoi chay Frontend trong cua so nay (foreground)
echo [+] Dang khoi chay Frontend (React + Vite)...
echo.
echo LUU Y: Nhan Ctrl+C de dung ca hai dich vu.
echo.
cd frontend
npm run dev

:: Neu nguoi dung thoat khoi npm run dev, script se dung lai o day
pause