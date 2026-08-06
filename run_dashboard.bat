@echo off
cd /d "%~dp0"
REM Port pinned HERE, not in .streamlit/config.toml -- that file is shared with
REM the Streamlit Cloud deploy, which manages its own port and fails its health
REM check if we pin one. 8504 keeps this off mlb-model's 8501 locally.
python -m streamlit run dashboard\app.py --server.port 8504
pause
