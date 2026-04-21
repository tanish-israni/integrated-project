@echo off
cd /d "C:\Users\tanis\Desktop\agentic ai\Agentic AI Project"
echo Starting Streamlit on http://127.0.0.1:8501
echo Keep this window open while using the project.
"C:\Program Files\Python311\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
pause
