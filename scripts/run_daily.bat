@echo off
chcp 65001 > nul
cd /d C:\Users\tatsu\noe-stock\scripts
echo [%date% %time%] Starting Noe Stock AI...

python fetch_data.py
python predict_today.py

echo [%date% %time%] Done.
