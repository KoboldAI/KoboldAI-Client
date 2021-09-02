@echo off
cd %~dp0
TITLE CMD for KoboldAI Runtime
SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder

:subfolder
umamba.exe install --no-shortcuts -r miniconda3 -n base -c conda-forge cython pip
call miniconda3\condabin\activate
cmd /k

:drivemap
subst K: miniconda3 >nul
call K:\python\condabin\activate
cmd /k