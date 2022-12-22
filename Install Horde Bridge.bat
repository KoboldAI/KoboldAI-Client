@echo off
cd /D %~dp0
TITLE KoboldAI - Installing Horde Bridge
SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder
IF %M%==3 GOTO drivemap_B

:subfolder
ECHO Runtime launching in subfolder mode
SET TEMP=%~DP0MINICONDA3
SET TMP=%~DP0MINICONDA3
call miniconda3\condabin\activate
git clone https://github.com/db0/KoboldAI-Horde-Bridge KoboldAI-Horde
python -m venv KoboldAI-Horde\venv
KoboldAI-Horde\venv\scripts\pip install -r KoboldAI-Horde\requirements.txt
cmd /k

:drivemap
ECHO Runtime launching in K: drive mode
subst /D K: >nul
subst K: miniconda3 >nul
SET TEMP=K:\
SET TMP=K:\
call K:\python\condabin\activate
git clone https://github.com/db0/KoboldAI-Horde-Bridge KoboldAI-Horde
python -m venv KoboldAI-Horde\venv
KoboldAI-Horde\venv\scripts\pip install -r KoboldAI-Horde\requirements.txt
cmd /k

:drivemap_B
ECHO Runtime launching in B: drive mode
subst /D B: >nul
subst B: miniconda3 >nul
SET TEMP=B:\
SET TMP=B:\
call B:\python\condabin\activate
git clone https://github.com/db0/KoboldAI-Horde-Bridge KoboldAI-Horde
python -m venv KoboldAI-Horde\venv
KoboldAI-Horde\venv\scripts\pip install -r KoboldAI-Horde\requirements.txt
cmd /k
