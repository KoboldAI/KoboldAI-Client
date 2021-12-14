@echo off
%~d0
cd %~dp0
TITLE KoboldAI - Updater
SET /P M=<loader.settings
IF %M%==1 GOTO drivemap
IF %M%==2 GOTO subfolder

:subfolder
call miniconda3\condabin\activate
GOTO GIT

:drivemap
subst K: miniconda3 >nul
call K:\python\condabin\activate
GOTO GIT

:GIT
if exist .git\ (
	git checkout -f
) else (
	git init     
	git remote add origin https://github.com/koboldai/koboldai-client    
	git fetch     
	git checkout main -f
)
cmd /k