#!/bin/bash

python scripts/main.py \
    --do predict \
    --predict_path "data/studies/*.nii.gz" \
    --root_dir . \
    --checkpoint_dir models/first \
    --result_dir results/first \
    --gin_param "PredictSpec.batch_size = 64"

python scripts/main.py \
    --do predict \
    --predict_path "results/first/*.nii.gz" \
    --root_dir . \
    --checkpoint_dir models/second \
    --result_dir results/second \
    --gin_param "PredictSpec.batch_size = 16"
