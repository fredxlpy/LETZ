# Code and Data for *Forget NLI, Use a Dictionary: Zero-Shot Topic Classification for Low-Resource Languages with Application to Luxembourgish*

All required libraries can found in `requirements.txt` and can be installed with `pip install -r requirements.txt`.

* `clean_data.py` extracts the relevant information from the raw LOD dataset
* `synonym_dataset.py` uses the cleaned LOD dataset and creates an NLI-like dataset with synoynms as "entailment" labels and randomly chosen words (based on a Levenstein-distance-based condition) as "contradiction" labels
* `translate_datasets.py` reformats and translates (German) datasets that we use for evaluation


To train models, use
```shell
python src/run_training.py \
  --model_type mbert luxembert \
  --training_data ours_hr ours_lr wnli xnli_hr_de xnli_lr_de \
  --seed 0 1 2 3
```

To evaluate the trained models, use
```shell
python src/run_evaluation.py \
  --model_type mbert \
  --training_data xnli_de_full \
  --evaluation_data gnad10_test_lb mlsum_test_lb agnews_test_lb yahoo_test_lb gnad10_test_de mlsum_test_de \
  --seed 0 1 2 3
```