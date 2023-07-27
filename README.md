# FPL_predictions

In this project we are fetching data from Fantasy Premier League and will use it to train three classification models, one each for result 1, X and 2.
We will use the models to score each teams next game. 

This project is created for me to learn more about different parts within ML-projects. Not focusing on doing best possible models. 
My personal goal with this project is to start using:
- mlflow
- docker
- Tensorflow
- organising a ML project with Python
- Setup enviroments
- Automate project

## Enviroment & libraries

Setup enviroment and install packages

```
conda create -p venv python==3.10 -y

conda activate venv/

conda install ipykernel -y

pip install -r requirements.txt
```

## run project with docker

Installed Docker (https://www.docker.com/get-started)

```
docker build --pull --rm -f "Dockerfile" -t fplpredictions:latest "."

docker run --rm -d  fplpredictions:latest
```

## Folder structure

-artifacts - data.
-- fetched_data - raw-data from API.
-- ml_result - result from latest run
-notebook - used for testing and analysing.
-src 
-- components - functions used in the project.
--- data - fetch data from API and transform data to train and score.
---- api - Functions that fetch the data.
---- transform - Functions used to create features.
---- ml - transform, train and score


## To do-list

- Improve the training-part.
- Divide training-part into several functions instead of one big.
- Document code and functions.
- Improve main.py as pipeline.
- Automate with Github actions
- Build airflow-flow just to learn airflow.
- don't re-run hold seasons data -> add data cumulative.