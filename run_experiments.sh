#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python linc/tests.py --filename_suffix neurosymbolic
CUDA_VISIBLE_DEVICES=0 python linc/tests.py --filename_suffix baseline --mode baseline