#!/bin/bash
conda env create -f environments/huggingface.yml
conda activate koboldai
python aiserver.py $*