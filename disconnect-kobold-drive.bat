@echo off
SET /P M=<loader.settings
IF %M%==3 subst /D B:
IF %M%==1 subst /D K:
cls
echo KoboldAI Drive disconnected
pause