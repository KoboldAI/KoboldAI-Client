@echo off
cd %~dp0
TITLE Jupyter for KoboldAI Runtime
SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder
IF %M%==3 GOTO drivemap_B

:Isolation
call conda deactivate 2>NUL
set Path=%windir%\system32;%windir%;C:\Windows\System32\Wbem;%windir%\System32\WindowsPowerShell\v1.0\;%windir%\System32\OpenSSH\
SET CONDA_SHLVL=
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=

:subfolder
umamba.exe install --no-shortcuts -r miniconda3 -n base -c conda-forge jupyterlab jupyterlab-git
call miniconda3\condabin\activate
jupyter-lab
cmd /k

:drivemap
subst K: miniconda3 >nul
umamba.exe install --no-shortcuts -r K:\python\ -n base -c conda-forge jupyterlab jupyterlab-git
call K:\python\condabin\activate
jupyter-lab
subst K: /D
cmd /k

:drivemap_B
subst B: miniconda3 >nul
umamba.exe install --no-shortcuts -r B:\python\ -n base -c conda-forge jupyterlab jupyterlab-git
call B:\python\condabin\activate
jupyter-lab
subst B: /D
cmd /k