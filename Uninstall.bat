@echo off 
cd /D %~dp0
TITLE KoboldAI Uninstall Helper
SET /P M=<loader.settings
IF %M%==3 subst /D B: >nul
IF %M%==1 subst /D K: >nul

IF "%1" == "FORCE" GOTO UNINSTALL

IF EXIST "Uninstall\unins000.exe" (
   start Uninstall\unins000.exe
   exit
) ELSE (
   echo This will remove all KoboldAI folders that do not contain user data.
   echo DO NOT CONTINUE IF KOBOLDAI IS NOT IN ITS OWN FOLDER! OTHERWISE YOUR OTHER DATA IN THIS FOLDER WILL BE DELETED AS WELL!
   pause
   set /P D=Type DELETE if you wish to continue the uninstallation: 
)

IF %D%==DELETE GOTO UNINSTALL
exit
	
:UNINSTALL
echo Uninstallation in progress, please wait...
set DM=Y
attrib -h .git >nul
for /d %%D in (*) do if not "%%~nxD"=="stories" if not "%%~nxD"=="userscripts" if not "%%~nxD"=="settings" if not "%%~nxD"=="softprompts" if not "%%~nxD"=="models" if not "%%~nxD"=="Uninstall" rmdir /S /Q %%~nxD
for %%i in (*) do if not "%%i"=="Uninstall.bat" del /q "%%i"
set /P DM=Would you like to delete the models folder? (Y/n) :
IF %DM%==Y rmdir models /s /q
IF %DM%==y rmdir models /s /q
set DM=N
set /P DM=Would you like to delete all other user folders? (y/N) :
IF %DM%==Y rmdir stories userscripts settings softprompts /s /q
IF %DM%==y rmdir stories userscripts settings softprompts /s /q
del Uninstall.bat