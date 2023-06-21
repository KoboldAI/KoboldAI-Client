@echo off
cd /D %~dp0

:Isolation
call conda deactivate 2>NUL
set Path=%windir%\system32;%windir%;C:\Windows\System32\Wbem;%windir%\System32\WindowsPowerShell\v1.0\;%windir%\System32\OpenSSH\
SET CONDA_SHLVL=
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=

TITLE KoboldAI - Git Transformers Installer
ECHO This script will replace the Transformers version with the latest Git Transformers which may contain breaking changes.
ECHO If you wish to return to the approved version of transformers you can run the install_requirements.bat script or KoboldAI Updater.
pause

SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder
IF %M%==3 GOTO drivemap_B

:subfolder
ECHO Runtime launching in subfolder mode
SET TEMP=%~DP0MINICONDA3
SET TMP=%~DP0MINICONDA3
call miniconda3\condabin\activate
pip install git+https://github.com/huggingface/transformers
cmd /k

:drivemap
ECHO Runtime launching in K: drive mode
subst /D K: >nul
subst K: miniconda3 >nul
SET TEMP=K:\
SET TMP=K:\
call K:\python\condabin\activate
pip install git+https://github.com/huggingface/transformers
cmd /k

:drivemap_B
ECHO Runtime launching in B: drive mode
subst /D B: >nul
subst B: miniconda3 >nul
SET TEMP=B:\
SET TMP=B:\
call B:\python\condabin\activate
pip install git+https://github.com/huggingface/transformers
cmd /k