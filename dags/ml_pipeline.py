

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

from ml_pipeline_functions import get_data, transform_data, setup_train_models, train, score

default_args = {
    'owner': 'Marcus Bergdahl',
    'start_date': datetime(2023, 9, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('ml_workflow', 
          default_args=default_args,
          schedule='0 0 * 9-5 4',
          catchup=False,
          max_active_runs=1
          )

get_data_task = PythonOperator(
    task_id='get_data',
    python_callable=get_data,
    dag=dag
)

transform_data_task = PythonOperator(
    task_id='transform_data',
    python_callable=transform_data,
    dag=dag
)

setup_train_models_task = PythonOperator(
    task_id='setup_train_models',
    python_callable=setup_train_models,
    dag=dag
)

train_1_task = PythonOperator(
    task_id='train_1',
    python_callable=train(label="label_1"),
    dag=dag
)

train_X_task = PythonOperator(
    task_id='train_X',
    python_callable=train(label="label_X"),
    dag=dag
)

train_2_task = PythonOperator(
    task_id='train_2',
    python_callable=train(label="label_2"),
    dag=dag
)

score_task = PythonOperator(
    task_id='score',
    python_callable=score,
    dag=dag
)

# Set task dependencies
get_data_task >> transform_data_task >> setup_train_models_task >> [train_1_task, train_X_task, train_2_task] >> score_task
