@echo off
%~d0
cd %~dp0
TITLE KoboldAI - Server
SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder

:subfolder
call miniconda3\condabin\activate
python aiserver.py %*
cmd /k

:drivemap
subst K: miniconda3 >nul
call K:\python\condabin\activate
python aiserver.py %*
subst K: /D
cmd /k
