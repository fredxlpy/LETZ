import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification
from datasets import Dataset
import evaluate
from tqdm import tqdm

# Clearing the GPU memory cache
torch.cuda.empty_cache()

# Setting the device to GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the evaluation metrics
acc = evaluate.load('accuracy')
f1 = evaluate.load('f1')
precision = evaluate.load('precision')
recall = evaluate.load('recall')

"""
model_type = 'luxembert'
training_data = 'ours_hr'
seed = 0
evaluation_data = 'sib-200'
"""


def evaluate(model_type, training_data, seed, evaluation_data):
    """
        Evaluate a text classification model on a given evaluation dataset.

        Args:
            model_type (str): The type of the model to evaluate. It can be either "mbert" or "luxembert".
            training_data (str): The name of the training dataset used to train the model.
            seed (int): The random seed used for reproducibility.
            evaluation_data (str): The name of the evaluation dataset to test the model on.

        Returns:
            accuracy (float): The accuracy score of the model on the evaluation dataset.
            f1_score (float): The macro F1 score of the model on the evaluation dataset.
            precision_score (float): The macro precision score of the model on the evaluation dataset.
            recall_score (float): The macro recall score of the model on the evaluation dataset.
        """

    # Validate the model type argument
    if model_type == "mbert":
        model_name = "bert-base-multilingual-cased"
    elif model_type == "luxembert":
        model_name = "lothritz/LuxemBERT"
    elif model_type == 'glot500':
        model_name = 'cis-lmu/glot500-base'
    else:
        raise ValueError("Model does not exist...")

    # Load evaluation data
    data = pd.read_excel(f'Data/eval_datasets/{evaluation_data}.xlsx')  # .sample(n=200)

    # Create a dictionary that maps each topic label to a numerical index
    topics_dict = {topic: i for i, topic in enumerate(data['label'].unique())}

    # Get the text samples from the data
    samples = data['text'].tolist()

    # Loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize and encode the text samples and the candidate topic labels as input pairs for the model
    inputs = tokenizer(np.repeat(samples, len(topics_dict)).tolist(),
                       [f"An dësem Beispill geet et em {label}." for label in topics_dict.keys()] * len(samples),
                       return_tensors="pt", padding="max_length", truncation='only_first',
                       max_length=512 if evaluation_data in ['gnad10_test_lb', 'lux_news_rtl'] else 128)

    # Convert the inputs to a Dataset object
    data_set = Dataset.from_dict(inputs.data)
    data_set = data_set.add_column('label', np.repeat(data['label'], len(topics_dict)).tolist())
    data_set.set_format("torch")

    model = torch.load(f'Models/{model_type}_{training_data}_{seed}', map_location=torch.device('cpu'))

    # Create a temporary output directory for saving the evaluation results
    output_dir = f"trainer/model_{model_type}_{training_data}_{evaluation_data}_{seed}"

    # Define hyper-parameters
    training_args = TrainingArguments(output_dir=output_dir,
                                      logging_dir='logs',
                                      per_device_eval_batch_size=32,
                                      do_train=False,
                                      do_eval=True,
                                      report_to="none")

    trainer = Trainer(model=model, args=training_args)

    # Predict the entailment scores for each input pair using the model
    if 'full' in training_data:
        entailment_scores = torch.tensor(trainer.predict(data_set).predictions).softmax(dim=-1)[:, 0]
    else:
        entailment_scores = torch.tensor(trainer.predict(data_set.remove_columns('label')).predictions).softmax(dim=-1)[:, 1]

    # Reshape the entailment scores into a matrix of shape (num_samples, num_topics)
    entailment_scores = entailment_scores.view(len(samples), len(topics_dict))

    # Get the indices of the topics with the highest entailment scores for each sample
    predictions = entailment_scores.argmax(dim=-1)

    # Replace the topic labels in the data with their numerical indices
    data['topic'] = data['label']
    data = data.replace({'label': topics_dict})

    # Compute and return the evaluation metrics using the predictions and the true labels
    accuracy = acc.compute(predictions=predictions, references=data['label'].tolist())['accuracy']
    f1_score = f1.compute(predictions=predictions, references=data['label'].tolist(), average='macro')['f1']
    precision_score = precision.compute(predictions=predictions, references=data['label'].tolist(), average='macro')['precision']
    recall_score = recall.compute(predictions=predictions, references=data['label'].tolist(), average='macro')['recall']

    full_f1_score = f1.compute(predictions=predictions, references=data['label'].tolist(), average=None)['f1']
    full_precision_score = precision.compute(predictions=predictions, references=data['label'].tolist(), average=None)['precision']
    full_recall_score = recall.compute(predictions=predictions, references=data['label'].tolist(), average=None)['recall']

    full_f1_score = {topic:score for topic,score in zip(list(topics_dict.keys()), full_f1_score)}
    full_precision_score = {topic: score for topic, score in zip(list(topics_dict.keys()), full_precision_score)}
    full_recall_score = {topic: score for topic, score in zip(list(topics_dict.keys()), full_recall_score)}

    return accuracy, f1_score, precision_score, recall_score


#from sklearn.metrics import f1_score, precision_score, recall_score, top_k_accuracy_score
#f1_score(data['label'].tolist(), predictions.numpy().tolist(), average=None)
#precision_score(data['label'].tolist(), predictions.numpy().tolist(), average=None)
#recall_score(data['label'].tolist(), predictions.numpy().tolist(), average=None)
#top_k_accuracy_score(data['label'].tolist(), entailment_scores, k=3)

