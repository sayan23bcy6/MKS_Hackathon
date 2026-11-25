@echo off
REM run_all.bat - single command for Windows
REM Usage: cd MKS_Hackathon && project\run_all.bat

echo Installing dependencies...
pip install -r project\app\requirements.txt

echo Running pipeline...
python project\run_pipeline.py

echo Running tests...
cd project
pytest tests\ -v
cd ..

echo Launching dashboard...
streamlit run project\app\dashboard.py

