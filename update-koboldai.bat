@echo off
cd /d %~dp0

:Isolation
call conda deactivate 2>NUL
set Path=%windir%\system32;%windir%;C:\Windows\System32\Wbem;%windir%\System32\WindowsPowerShell\v1.0\;%windir%\System32\OpenSSH\
SET CONDA_SHLVL=
SET PYTHONNOUSERSITE=1
SET PYTHONPATH=

TITLE KoboldAI - Updater
SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder
IF %M%==3 GOTO drivemap_B

:subfolder
call miniconda3\condabin\activate
GOTO GIT

:drivemap
subst /D K: >nul
subst K: miniconda3 >nul
call K:\python\condabin\activate
GOTO GIT

:drivemap_B
subst /D B: >nul
subst B: miniconda3 >nul
call B:\python\condabin\activate
GOTO GIT

:GIT
ECHO 1. KoboldAI Main (The Official stable version of KoboldAI)
ECHO 2. KoboldAI United (Development Version, new features but may break at any time)
SET /P V=Enter your desired version or type your own GIT URL:
IF %V%==1 (
SET origin=https://github.com/koboldai/koboldai-client
SET branch=main
) ELSE (
	IF %V%==2 (
		SET origin=https://github.com/henk717/koboldai
		SET branch=united
	) ELSE (
		SET origin=%v%
		SET /P branch=Specify the GIT Branch:
	)
)

git init     
git remote remove origin
git remote add origin %origin%    
git fetch --all
git checkout %branch% -f
git reset --hard origin/%branch%
git submodule update --init --recursive
IF %M%==1 umamba.exe install --no-shortcuts -r K:\python\ -n base -f "%~dp0\environments\huggingface.yml" -y --always-copy
IF %M%==2 umamba.exe install --no-shortcuts -r miniconda3 -n base -f environments\huggingface.yml -y --always-copy
IF %M%==3 umamba.exe install --no-shortcuts -r B:\python\ -n base -f "%~dp0\environments\huggingface.yml" -y --always-copy


%windir%\system32\timeout -t 10