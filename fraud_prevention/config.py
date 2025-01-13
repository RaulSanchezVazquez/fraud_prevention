#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings

# Supress warnings
warnings.filterwarnings("ignore")

# The package root directory
PRJ_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-2])

# Scaffolding local folders
for folder in [
        'data/raw', 'data/interim', 'data/external', 'data/processed',
        'models']:

    os.makedirs(
        os.path.join(PRJ_DIR, folder),
        exist_ok=True)
