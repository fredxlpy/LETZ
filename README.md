# Code and Data for *Forget NLI, Use a Dictionary: Zero-Shot Topic Classification for Low-Resource Languages with Application to Luxembourgish*

## Code
All required libraries can found in `requirements.txt` and can be installed with `pip install -r requirements.txt`.

* `src/process_LOD.py` extracts the relevant information from the raw LOD dataset
* `src/synonym_dataset.py` uses the processed LOD data to create the ```LETZ-SYN``` dataset
* `src/translation_dataset.py` uses the processed LOD data to create the ```LETZ-WoT``` dataset


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
  --model_type mbert luxembert \
  --training_data ours_hr ours_lr wnli xnli_hr_de xnli_lr_de \
  --evaluation_data lux_news_rtl sib-200 \
  --seed 0 1 2 3
```

## Data
The original [Luxembourg Online Dictionary](https://lod.lu) (LOD) data can be downloaded from the [Luxembourgish Open Data Platform](https://data.public.lu/fr/datasets/letzebuerger-online-dictionnaire-lod-linguistesch-daten/) or can be accessed via their [API](https://data.public.lu/fr/datasets/letzebuerger-online-dictionnaire-lod-public-api/). All of their data is available under a [Creative Commons Zero (CC0)](https://creativecommons.org/publicdomain/zero/1.0/) license.
