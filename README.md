# Code for EMNLP industry track paper

All requirements to run the code can be found in `requirements.txt` and all required packages can be installed with `pip install -r requirements.txt`.

* `clean_data.py` extracts the relevant information from the raw LOD dataset
* `synonym_dataset.py` uses the cleaned LOD dataset and creates an NLI-like dataset with synoynms as "entailment" labels and randomly chosen words (based on a Levenstein-distance-based condition) as "contradiction" labels
* `translate_datasets.py` reformats and translates (German) datasets that we use for evaluation
