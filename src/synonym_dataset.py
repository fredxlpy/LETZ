import pandas as pd
import numpy as np
from Levenshtein import distance as lev
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split

random.seed(10)

data = pd.read_excel('Data/LOD_clean.xlsx')

data = data.dropna(subset=['Synonyms', 'examples'])
data = data[data['partOfSpeech']=='SUBST']

new_data = pd.DataFrame()
row_counter = 0
for i, row in data.iterrows():

    for synonym in row['Synonyms'].split(', '):

        for example in row['examples'].split('; '):

            new_data.loc[row_counter, 'word'] = row['lemma']
            new_data.loc[row_counter, 'Synonym'] = synonym
            new_data.loc[row_counter, 'lvd'] = lev(row['lemma'], synonym)
            new_data.loc[row_counter, 'example'] = example

            row_counter +=1

# Remove samples where the word itself is too similar to its synonym
new_data = new_data[new_data['lvd']>=3]

# Choose contradictory words
vocabulary = np.unique(new_data['word']).tolist()
for i, row in new_data.iterrows():

    contradiction_label = random.choice(vocabulary)
    while any([lev(contradiction_label, word)<3 for word in row['example'].split(' ')]):
        contradiction_label = random.choice(vocabulary)
    new_data.loc[i, 'contradiction_word'] = contradiction_label

# Create NLI dataset from synonyms and unrelated (contradictory) words
nli_dataset = pd.DataFrame(columns=['text', 'label', 'class'])
for i, row in tqdm(new_data.iterrows()):
    nli_dataset = pd.concat([nli_dataset,
                             pd.DataFrame({'text': [row['example']], 'label': [row['Synonym']], 'class': [1]})])

    nli_dataset = pd.concat([nli_dataset,
                             pd.DataFrame({'text': [row['example']], 'label': [row['contradiction_word']], 'class': [0]})])

train_set, val_set = train_test_split(nli_dataset, train_size=0.8, shuffle=True, stratify=nli_dataset['class'])
val_set, test_set = train_test_split(val_set, train_size=0.5, shuffle=True, stratify=val_set['class'])

train_set.to_excel('Data/LETZ-SYN/train.xlsx', index=False)
val_set.to_excel('Data/LETZ-SYN/val.xlsx', index=False)
test_set.to_excel('Data/LETZ-SYN/test.xlsx', index=False)