from transformers import AutoTokenizer, Trainer, AutoModelForSequenceClassification, TrainingArguments
from datasets import load_dataset
import pandas as pd
import torch
import shutil

# Importing custom functions for data preprocessing and metrics computation
try:
    from src.utils import compute_metrics, preprocess_ours, preprocess_wnli, preprocess_xnli
except ImportError:
    from utils import compute_metrics, preprocess_ours, preprocess_wnli, preprocess_xnli

# Clearing the GPU memory cache
torch.cuda.empty_cache()

# Setting the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
model_type = 'luxembert'
training_data = 'ours_lr'
seed = 1
"""


def train(model_type, training_data, seed):
    """
    A function to train a sequence pair classification model on a given dataset using a given model type and seed.

    Args:
        model_type (str): The type of the model to use. Either "mbert" for multilingual BERT or "luxembert" for LuxemBERT.
        training_data (str): The name of the dataset to use for training (e.g., "ours_hr", "wnli", "xnli_lr_de").
        seed (int): The random seed to use for reproducibility.

    Returns:
        model (AutoModelForSequenceClassification): The fine-tuned model.
    """

    # Initializing global variables for train and validation data
    global train_data, val_data

    # Setting the model name based on the model type
    if model_type == "mbert":
        model_name = "bert-base-multilingual-cased"
    elif model_type == "luxembert":
        model_name = "lothritz/LuxemBERT"
    else:
        raise ValueError('Model does not exist...')

    # Loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def model_init():

        # Load base model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2)

        # Move the model to the device (GPU or CPU)
        model.to(device)

        return model

    # Setting the number of samples for train and validation
    if 'wnli' in training_data or 'lr' in training_data:
        n_train = 568
        n_val = 63
    else:
        n_train = 11822
        n_val = 1478

    # Load and tokenize data
    if 'ours_trans' in training_data:
        train_data = pd.read_excel('Data/Ours/translation_dataset/train.xlsx')
        train_data = preprocess_ours(train_data, tokenizer, n=n_train)

        val_data = pd.read_excel('Data/Ours/translation_dataset/val.xlsx')
        val_data = preprocess_ours(val_data, tokenizer, n=n_val)

    elif 'ours' in training_data:
        train_data = pd.read_excel('Data/Ours/synonym_dataset/train.xlsx')
        train_data = preprocess_ours(train_data, tokenizer, n=n_train)

        val_data = pd.read_excel('Data/Ours/synonym_dataset/val.xlsx')
        val_data = preprocess_ours(val_data, tokenizer, n=n_val)

    elif training_data == 'wnli':
        train_data = pd.read_csv("https://raw.githubusercontent.com/Trustworthy-Software/LuxemBERT/main/L-WNLI/train_l.csv")
        train_data = preprocess_wnli(train_data, tokenizer, n=n_train)

        val_data = pd.read_csv("https://raw.githubusercontent.com/Trustworthy-Software/LuxemBERT/main/L-WNLI/dev_l.csv")
        val_data = preprocess_wnli(val_data, tokenizer, n=n_val)

    elif 'xnli' in training_data:
        lang = training_data[-2:]
        train_data = load_dataset("xnli", lang, split='train')
        train_data = preprocess_xnli(train_data, tokenizer, n=n_train)

        val_data = load_dataset("xnli", lang, split='test')  # We use the test data here for validation (more samples)
        val_data = preprocess_xnli(val_data, tokenizer, n=n_val)

    else:
        raise ValueError('Dataset does not exist...')

    # Setting the output directory name
    output_dir = f"trainer/model_{model_type}_{training_data}_{seed}"

    # Removing any existing checkpoints in the output directory
    shutil.rmtree(output_dir, ignore_errors=True)

    # Define hyper-parameters
    training_args = TrainingArguments(output_dir=output_dir,
                                      max_steps=-1,
                                      num_train_epochs=5,

                                      evaluation_strategy="epoch",

                                      save_strategy="epoch",
                                      save_total_limit=1,

                                      logging_strategy='steps',
                                      logging_steps=5,
                                      logging_first_step=True,
                                      logging_dir='logs',

                                      learning_rate=2e-5,
                                      per_device_train_batch_size=32,
                                      gradient_accumulation_steps=1,
                                      per_device_eval_batch_size=32,

                                      seed=seed if seed is not None else 42,
                                      warmup_ratio=0.1,
                                      do_train=True,
                                      do_eval=True,
                                      report_to="none",
                                      optim='adamw_torch',

                                      load_best_model_at_end=True,
                                      metric_for_best_model='eval_loss')

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    return trainer.model