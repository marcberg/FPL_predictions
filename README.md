# FPL_predictions

In this project we are fetching data from Fantasy Premier League and will use it to train three classification models, one each for result 1, X and 2.
We will use the models to score each teams next game. 

This project is created for me to learn more about different parts within ML-projects. Not focusing on doing best possible models. 
My personal goal with this project is to start using:
- mlflow
- Docker
- Tensorflow
- Organising a ML project with Python
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

Installed Docker https://www.docker.com/get-started

```
docker build -t fplpredictions .

docker run -d --name fpl_predictions_container fplpredictions:latest
```

## Folder structure

```
FPL_predictions/
│
├─ artifacts/
│   ├─ fetched_data/
│   └─ ml_result/
│
├─ notebook/
│
└─ src/
    ├─ components/
    │   ├─ data/
    │   │   ├─ api/
    │   │   └─ transform/
    │   └─ ml/
```


### FPL_predictions

Includes main.py and main.ipynb. The later is used for testing main.
There could be som test_*.ipynb files which have been used for testing. 

Other files are Dockerfile, .gitignore, LICENSE and requirements.txt. 


### Artifacts

- artifacts: Transformed data divided into train, test, validation and score. 
- fetched_data: Fetched data by the API.
- ml_results: Results (metrics, grid, feature importance) from the models from the latest run.  


### notebook

.ipynb-files used for EDA and testing and understanding of models. 


### src

src includes all  functions used in the project.

- src: utils.py, includes general function used all over the project. 
- components: divide data function from ml functions.
- data: has fetch_data.py from fetching data with API, and transform_data which transformed the fetched data.
- api: .py-files with functions used for fetching the data. Divided in to categories.
- transform: .py-files with functions used for transformed the fetched data. Divided in to categories.
- ml: .py-files with functions performing all ML-related activities. Like splitting traing, test and val, hyperparameter-tuning, evalutation and scoring.


## To do-list

- Improve the training-part.
- Document code and functions.
- Improve main.py as pipeline.
- Automate with Github actions
- Build airflow-flow just to learn airflow.