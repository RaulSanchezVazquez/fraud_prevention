#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import warnings

# Supress warnings
warnings.filterwarnings("ignore")

# The package root directory
PRJ_DIR = '/'.join(os.path.abspath(__file__).split('/')[:-2])

# Scaffolding local folders
for folder in ['raw', 'interim', 'processed', 'external']:
    os.makedirs(
        os.path.join(PRJ_DIR, folder),
        exist_ok=True)
