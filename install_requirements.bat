@echo off
title KoboldAI Runtime Installer (MicroMamba)
echo Please choose one of the following transformers options
echo 1. Official Transformers (Recommended)
echo 2. Finetune Transformers (For old 6B models)
echo.
echo Errors? Rerun this as admin so it can add the needed LongPathsEnabled registery tweak.
echo Installer failed or crashed? Run it again so it can continue.
echo Only Windows 10 and higher officially supported, older Windows installations can't handle the paths.
echo.

SET /P B=Type the number of the desired option and then press ENTER: 

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul
%~d0
cd %~dp0

if exist miniconda3\ (
  echo Delete existing installation?
  echo This is required if you are switching modes, or if you get dependency errors in the game.
  echo 1. Yes
  echo 2. No
  SET /P D=Type the number of the desired option and then press ENTER: 
) ELSE (
	SET D=Workaround
)
IF %D%==1 rmdir /s /q miniconda3

:Mode
echo Which installation mode would you like?
echo 1. Temporary Drive Letter (Mounts the folder as drive K:, more stable and portable)
echo 2. Subfolder (Traditional method, can't run in folder paths that contain spaces)
echo.
SET /P M=Type the number of the desired option and then press ENTER: 
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder
ECHO Incorrect choice
GOTO MODE


:drivemap
echo 1 > loader.settings
subst K: /D >nul
mkdir miniconda3 
subst K: miniconda3
copy umamba.exe K:\umamba.exe
K:
umamba.exe create -r K:\python\ -n base
IF %B%==1 umamba.exe install --no-shortcuts -r K:\python\ -n base -f "%~dp0\environments\huggingface.yml" -y
IF %B%==2 umamba.exe install --no-shortcuts -r K:\python\ -n base -f "%~dp0\environments\finetuneanon.yml" -y
umamba.exe -r K:\ clean -a -y
subst K: /d
pause
exit

:subfolder
echo 2 > loader.settings
umamba.exe create -r miniconda3\ -n base
IF %B%==1 umamba.exe install --no-shortcuts -r miniconda3 -n base -f environments\huggingface.yml -y
IF %B%==2 umamba.exe install --no-shortcuts -r miniconda3 -n base -f environments\finetuneanon.yml -y
umamba.exe clean -a -y
pause
exit
