@echo off
cd /d "D:\d\rdkit files\Easy_rdkit"
call venv\Scripts\activate
pip install requests
pip install pubchempy
pip install flask flask-cors rdkit pandas requests
python app.py
pause