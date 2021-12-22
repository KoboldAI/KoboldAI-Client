@echo off
cd /D %~dp0
TITLE KoboldAI - Server
SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder

:subfolder
ECHO Runtime launching in subfolder mode
SET TEMP=%~DP0MINICONDA3
SET TMP=%~DP0MINICONDA3
call miniconda3\condabin\activate
python aiserver.py %*
cmd /k

:drivemap
ECHO Runtime launching in K: drive mode
subst K: miniconda3 >nul
SET TEMP=K:\
SET TMP=K:\
call K:\python\condabin\activate
python aiserver.py %*
subst K: /D
cmd /k
