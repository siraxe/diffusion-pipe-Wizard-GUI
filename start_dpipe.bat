@echo off
echo Starting WSL and running Python application...

REM Start WSL, activate conda environment and run Python script
wsl -d Ubuntu -e bash -c "cd /home/username/Dpipe && source ~/miniconda3/bin/activate && source $(conda info --base)/etc/profile.d/conda.sh && conda activate ./dp_env && python3 -m flet_app.flet_app"

echo WSL session ended.
pause