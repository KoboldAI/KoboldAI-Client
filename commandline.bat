@echo off
cd /D %~dp0

:Isolation
call conda deactivate 2>NUL
set Path=%windir%\system32;%windir%;C:\Windows\System32\Wbem;%windir%\System32\WindowsPowerShell\v1.0\;%windir%\System32\OpenSSH\
SET CONDA_SHLVL=
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=

TITLE CMD for KoboldAI Runtime
SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder
IF %M%==3 GOTO drivemap_B

:subfolder
call miniconda3\condabin\activate
cmd /k "%*"

:drivemap
subst K: miniconda3 >nul
call K:\python\condabin\activate
cmd /k "%*"

:drivemap_B
subst B: miniconda3 >nul
call B:\python\condabin\activate
cmd /k "%*"