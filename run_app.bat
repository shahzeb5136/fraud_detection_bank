@echo off
cd /d "%~dp0"
if not exist artifacts\models.pkl (
  echo Building the demo (about 90 seconds)...
  python -m fraudlab.build
)
streamlit run app.py
