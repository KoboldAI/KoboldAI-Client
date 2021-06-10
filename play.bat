@echo off
TITLE KoboldAI - Client
call miniconda3\condabin\activate koboldai
cls
python aiserver.py
cmd /k
