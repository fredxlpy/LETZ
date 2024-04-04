import argparse
import pandas as pd

try:
    from src.evaluation import evaluate
except ImportError:
    from evaluation import evaluate

# Create an argument parser
parser = argparse.ArgumentParser()
# Add arguments
parser.add_argument("--model_type", type=str, nargs='+')
parser.add_argument("--training_data", type=str, nargs='+')
parser.add_argument("--evaluation_data", type=str, nargs='+')
parser.add_argument("--seed", type=int, default=[0], nargs='+')

args = parser.parse_args()

"""
MODEL_TYPE = 'luxembert'
TRAINING_DATA = 'ours_hr'
EVALUATION_DATA = 'lux_news_rtl'
SEED = 0
"""

results = []

for MODEL_TYPE in args.model_type:

    for TRAINING_DATA in args.training_data:

        for EVALUATION_DATA in args.evaluation_data:

            for SEED in args.seed:

                if 'xnli' in TRAINING_DATA and MODEL_TYPE == 'luxembert':
                    continue

                accuracy, f1_score, precision, recall = evaluate(
                    model_type=MODEL_TYPE,
                    training_data=TRAINING_DATA,
                    evaluation_data=EVALUATION_DATA,
                    seed=SEED)

                print('Model:', MODEL_TYPE)
                print(f'Fine-tuned on', TRAINING_DATA)
                print('Evaluated on', EVALUATION_DATA)
                print('Accuracy:', accuracy)
                print('F1 Score:', f1_score)

                results.append([
                    MODEL_TYPE, TRAINING_DATA, EVALUATION_DATA, accuracy, f1_score, precision, recall, SEED
                ])

                pd.DataFrame(
                    results, columns=['model', 'train_set', 'eval_set', 'accuracy', 'f1_score', 'precision', 'recall', 'seed']
                ).to_excel(f'Output/results_{SEED}_{MODEL_TYPE}.xlsx', index=False)