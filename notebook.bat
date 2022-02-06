@echo off
cd %~dp0
TITLE Jupyter for KoboldAI Runtime
SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder
IF %M%==3 GOTO drivemap_B

:subfolder
umamba.exe install --no-shortcuts -r miniconda3 -n base -c conda-forge jupyter
call miniconda3\condabin\activate
jupyter notebook
cmd /k

:drivemap
subst K: miniconda3 >nul
umamba.exe install --no-shortcuts -r K:\python\ -n base -c conda-forge jupyter
call K:\python\condabin\activate
jupyter notebook
subst K: /D
cmd /k

:drivemap_B
subst B: miniconda3 >nul
umamba.exe install --no-shortcuts -r B:\python\ -n base -c conda-forge jupyter
call B:\python\condabin\activate
jupyter notebook
subst B: /D
cmd /k