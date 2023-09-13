from dags.ml_pipeline_functions import get_data, transform_data, setup_train_models, train, score

if __name__ == "__main__":
    get_data()
    transform_data()
    setup_train_models()
    train(label="label_1")
    train(label="label_X")
    train(label="label_2")
    score()