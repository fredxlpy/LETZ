import evaluate
import numpy as np
from datasets import Dataset

# Load evaluation metric
acc = evaluate.load('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return acc.compute(predictions=predictions, references=labels)


def preprocess_ours(data, tokenizer, n):
    def tokenize_function(examples):
        return tokenizer(examples["text"], [f"An dësem Beispill geet et em {label}." for label in examples["label"]],
                         padding="max_length", max_length=128, truncation='only_first')

    dataset = data.sample(n=n)
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.map(tokenize_function, batched=True).shuffle(True)
    dataset = dataset.remove_columns(['text', 'label'])
    dataset = dataset.rename_column('class', 'label')

    return dataset


def preprocess_wnli(data, tokenizer, n):
    def tokenize_function(examples):
        return tokenizer(examples["sentence1"], examples["sentence2"],
                         padding="max_length", max_length=128, truncation='only_first')

    dataset = data.sample(n=n)
    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.map(tokenize_function, batched=True).shuffle(True)

    return dataset


def preprocess_xnli(data, tokenizer, n):
    def tokenize_function(examples):
        return tokenizer(examples["premise"], examples["hypothesis"],
                         padding="max_length", max_length=128, truncation='only_first')

    dataset = data.filter(lambda x: x['label'] != 1)
    dataset = Dataset.from_pandas(dataset.to_pandas().replace({'label': {0: 1, 2: 0}}))
    dataset = dataset.class_encode_column('label')
    dataset = dataset.train_test_split(train_size=n, stratify_by_column='label', seed=None)['train']
    dataset = dataset.map(tokenize_function, batched=True).shuffle(True)

    return dataset