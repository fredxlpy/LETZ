import argparse
import torch
try:
    from src.training import train
except ImportError:
    from training import train

# Create an argument parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument("--model_type", type=str, nargs='+')
parser.add_argument("--training_data", type=str, nargs='+')
parser.add_argument("--seed", type=int, default=[0], nargs='+')

# Parse the arguments and store them in a variable
args = parser.parse_args()

"""
MODEL_TYPE = 'luxembert'
TRAINING_DATA = 'ours_lr'
SEED = 0
"""

# Loop over the possible values of model type, training data and seed
for MODEL_TYPE in args.model_type:

    for TRAINING_DATA in args.training_data:

        for SEED in args.seed:

            # Skip the case where we try to fine-tune a monolingual model on a multilingual data
            if 'xnli' in TRAINING_DATA and MODEL_TYPE == 'luxembert':
                continue

            model = train(
                model_type=MODEL_TYPE,
                training_data=TRAINING_DATA,
                seed=SEED)

            torch.save(model, f'./Models/{MODEL_TYPE}_{TRAINING_DATA}_{SEED}')
