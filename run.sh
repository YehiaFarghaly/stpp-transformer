#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Train the model
python main.py --dataset Earthquake --timesteps 500

# test the model
python test.py --dataset Earthquake --ckpt_path best_transformer_model.ckpt
