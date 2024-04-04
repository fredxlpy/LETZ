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

for lang in ['fr', 'de', 'en', 'pt']:

    lang_data = data.dropna(subset=[lang])

    for i, row in tqdm(lang_data.iterrows(), total=len(lang_data)):

        for translation in row[lang].split(', '):

            for example in row['examples'].split('; '):

                translation = translation.replace("(", "")
                translation = translation.replace(")", "")

                new_data.loc[row_counter, 'word'] = row['lemma']
                new_data.loc[row_counter, 'translation'] = translation
                new_data.loc[row_counter, 'lvd'] = lev(row['lemma'].lower(), translation.lower())
                new_data.loc[row_counter, 'example'] = example
                new_data.loc[row_counter, 'lang'] = lang

                row_counter +=1

# Remove samples where the word itself is too similar to its synonym (based on Levenshtein distance)
new_data = new_data[new_data['lvd']>=3]

# Choose contradictory words
for lang in ['fr', 'de', 'en', 'pt']:

    vocabulary = np.unique(new_data.loc[new_data['lang']==lang, 'translation']).tolist()

    for i, row in new_data[new_data['lang']==lang].iterrows():

        contradiction_label = random.choice(vocabulary)
        while any([lev(contradiction_label, word)<3 for word in row['example'].split(' ')]) and contradiction_label==row['translation']:
            contradiction_label = random.choice(vocabulary)
        new_data.loc[i, 'contradiction_word'] = contradiction_label

# Create NLI dataset from synonyms and unrelated (contradictory) words
nli_dataset = pd.DataFrame(columns=['text', 'label', 'class'])
for i, row in tqdm(new_data.iterrows()):
    nli_dataset = pd.concat([nli_dataset,
                             pd.DataFrame({'text': [row['example']], 'label': [row['translation']], 'class': [1], 'lang': row['lang']})])

    nli_dataset = pd.concat([nli_dataset,
                             pd.DataFrame({'text': [row['example']], 'label': [row['contradiction_word']], 'class': [0], 'lang': row['lang']})])

train_set, val_set = train_test_split(nli_dataset, train_size=0.8, shuffle=True, stratify=nli_dataset['class'])
val_set, test_set = train_test_split(val_set, train_size=0.5, shuffle=True, stratify=val_set['class'])

train_set.to_excel('eval_datasets/LETZ-WoT/train.xlsx', index=False)
val_set.to_excel('eval_datasets/LETZ-WoT/val.xlsx', index=False)
test_set.to_excel('eval_datasets/LETZ-WoT/test.xlsx', index=False)