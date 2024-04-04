import pandas as pd
from tqdm import tqdm

data = pd.read_json('./Data/new_lod-art.json')
new_data = pd.DataFrame()

row_counter = 0
vulgs, pejs, fams = [], [], []

# Loop over vocabulary
for i, row in tqdm(data.iterrows()):

    try:
        meanings = row['microStructures'][0]['grammaticalUnits'][0]['meanings']

        if len(meanings) > 1:
            meaning_ids = [meaning['number'] for meaning in meanings]
        else:
            meaning_ids = [1]

        # Loop over different meanings of given word
        for meaning_id, meaning in zip(meaning_ids, meanings):

            row_counter +=1

            for col in ['lod_id', 'partOfSpeech', 'partOfSpeechLabel', 'lemma']:
                new_data.loc[row_counter, col] = data.loc[i, col]

            # ===================== #
            # Retrieve translations #
            # ===================== #
            for lang in ['fr', 'de', 'en', 'pt']:

                try:
                    new_data.loc[row_counter, lang] = ', '.join([trans['content'] for trans in meaning['targetLanguages'][lang]['parts'] if trans['type']=='translation'])
                except:
                    pass

            # ========================== #
            # Retrieve example sentences #
            # ========================== #
            try:
                examples = []
                for example in meaning['examples']:
                    doc = ' '.join([word['content'] for word in example['parts'][0]['parts']])
                    if 'VULG' in doc or 'VULG' in meaning['attributes']:
                        vulgs.append(doc)
                    if 'PEJ' in doc or 'PEJ' in meaning['attributes']:
                        pejs.append(doc)
                    if 'FAM' in doc or 'FAM' in meaning['attributes']:
                        fams.append(doc)
                    # Remove indicator from string
                    doc = ' '.join([word for word in doc.split(' ') if word not in [
                        'VULG','EGS','FAM','GEHUEW','KANNERSPROOCH','NEOL','PEJ','VEREELZT']])
                    # Uppercase first letter of every example
                    doc = doc[0].upper() + doc[1:]
                    # Add full-stop to the end if string does not end with punctuation
                    doc = doc + '.' if doc[-1] not in ['.','?','!'] else doc
                    # Remove spaces
                    doc = doc.replace("\' ", "\'")
                    doc = doc.replace(" ,", ",")
                    doc = doc.replace(" .", ".")
                    doc = doc.replace(" !", "!")
                    doc = doc.replace(" ?", "?")
                    examples.append(doc)

                new_data.loc[row_counter, 'examples'] = '; '.join(examples)
            except:
                pass

            # ================= #
            # Retrieve synonyms #
            # ================= #
            try:
                synonym_groups = row['allSynonyms']['synonymGroups']

                # Case: Word has more than one meaning
                if len(meanings)>1:
                    try:
                        # Select the synonym group for the given meaning (or meaning_id)
                        synonym_group = [group for group in synonym_groups if group['number']==meaning_id][0]
                        # Extract the synonyms from the given synonym group
                        synonyms = [synonym['syn'] for synonym in synonym_group['toSynonyms']]
                        # Add the synonyms to the dataset
                        new_data.loc[row_counter, 'Synonyms'] = ', '.join(synonyms)
                    except:
                        pass

                # Case: Word has only a single meaning
                else:
                    try:
                        # Extract the synonyms of the given word
                        synonyms = [synonym['syn'] for synonym in synonym_groups[0]['toSynonyms']]
                        # Add the synonyms to the dataset
                        new_data.loc[row_counter, 'Synonyms'] = ', '.join(synonyms)
                    except:
                        pass


            except:
                pass

    except:
        pass

new_data.to_excel('./Data/LOD_clean.xlsx', index=False)