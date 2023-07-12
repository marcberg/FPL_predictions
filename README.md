# FPL_predictions

## Enviroment & libraries

Setup enviroment

```
conda create -p venv python==3.10 -y

conda activate venv/

conda install ipykernel -y

```

Install packages needed

```
pip install -r requirements.txt
```

## Folder structure

artifacts - data.
    - fetched_data - raw-data from API.

notebook - used for testing and analysing.

src 
    - components - functions used in the project.
        - data - fetch data from API and transform data to train and score.
            - api - Functions that fetch the data.
            - transform - Functions used to create features.
        - ml - transform, train and score


## To do-list

- Improve the training-part.
- overtraining a lot

## How to run all project as-is 2023-07-03 (to be put in pipeline)

1. Fetch data from API - src/compontents/data/fetch_data.py
2. Create data for ML - src/compontents/data/transform_data.py
3. Split data into train, test, val and score. Train model with hyperparameter tuning - src/compontents/ml/data_ingest_transform_train.py
4. Score - src/compontents/ml/score.py

or run "main.py"/"main.ipynb"

## To be added

- Docker
- Automate
    - airflow? but then I can't turn of my computer.
    - github actions? 