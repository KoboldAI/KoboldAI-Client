@echo off
title Installing Portable Python (Miniconda3)
echo Miniconda3's installer will overwrite existing Miniconda3 shortcuts in the startmenu (We currently can not prevent this)
echo Please choose one of the following transformers options
echo 1. Finetuneanon Transformers
echo 2. Official Transformers (Only use this if your model does not support half)
echo.
echo Errors? Rerun this as admin so it can add the needed registery tweak.
echo.

SET /P M=Type the number of the desired option and then press ENTER: 

Reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v "LongPathsEnabled" /t REG_DWORD /d "1" /f 2>nul

cd %~dp0
rmdir /s /q miniconda3
where /q curl.exe
IF ERRORLEVEL 1 (
    bitsadmin /transfer miniconda /download /priority normal https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe "%~dp0\miniconda3.exe"
) ELSE (
    curl -o miniconda3.exe https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
)
miniconda3.exe /S /InstallationType=JustMe /RegisterPython=0 /AddTopath=0 /NoScripts=1 /NoRegistry=1 /D=%~dp0\miniconda3
del miniconda3.exe
call miniconda3\condabin\activate
call conda install --all --no-shortcuts -y git pytorch tensorflow-gpu colorama Flask-SocketIO -c pytorch -c conda-forge
IF %M%==1 pip install git+https://github.com/finetuneanon/transformers@gpt-neo-localattention3
IF %M%==2 call conda install --no-shortcuts -y transformers -c huggingface
call conda clean -a -y
echo All done!
pause
