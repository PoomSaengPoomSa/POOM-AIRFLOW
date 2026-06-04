from datetime import datetime, timedelta
import pendulum
from airflow import DAG  # type: ignore
from airflow.operators.bash import BashOperator  # type: ignore

# 한국 시간대(KST) 정의
kst = pendulum.timezone("Asia/Seoul")

default_args = {
    "owner": "poom_ai_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 5, 28, tzinfo=kst),
    "email": ["admin@poom.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "ml_integrated_training_pipeline",
    default_args=default_args,
    description="Integrated Economic Indicators MLflow Parallel Training Pipeline on S3 Paths",
    schedule="0 4 1 * *",                   # 매월 1일 오전 4시 정각 기동 (요구에 맞춰 스케줄 변경 가능)
    catchup=False,
    tags=["poom", "ml", "integrated", "parallel", "mlflow"],
) as dag:

    # =========================================================================
    # 1. GOLD MODEL PIPELINE (금값 예측 모델)
    # =========================================================================
    gold_dir = "/opt/airflow/dags/scripts/ml/gold"

    gold_get_data = BashOperator(
        task_id="gold_get_data",
        bash_command=f"cd {gold_dir} && mkdir -p data models && PYTHONPATH=. python3 utils/get_data.py"
    )

    gold_preprocess = BashOperator(
        task_id="gold_preprocess",
        bash_command=f"cd {gold_dir} && mkdir -p data models && PYTHONPATH=. python3 utils/preprocess.py"
    )

    gold_train = BashOperator(
        task_id="gold_train",
        bash_command=f"cd {gold_dir} && mkdir -p data models && PYTHONPATH=. python3 train.py"
    )

    gold_test = BashOperator(
        task_id="gold_test",
        bash_command=f"cd {gold_dir} && mkdir -p data models && PYTHONPATH=. python3 test.py"
    )

    gold_explain = BashOperator(
        task_id="gold_explain",
        bash_command=f"cd {gold_dir} && mkdir -p data models && PYTHONPATH=. python3 explain.py"
    )

    gold_interpret = BashOperator(
        task_id="gold_interpret",
        bash_command=f"cd {gold_dir} && mkdir -p data models && PYTHONPATH=. python3 interpret_xai.py"
    )

    gold_get_data >> gold_preprocess >> gold_train >> gold_test >> gold_explain >> gold_interpret

    # =========================================================================
    # 2. BASE RATE MODEL PIPELINE (기준금리 예측 모델)
    # =========================================================================
    base_rate_dir = "/opt/airflow/dags/scripts/ml/base_rate"

    base_rate_get_data = BashOperator(
        task_id="base_rate_get_data",
        bash_command=f"cd {base_rate_dir} && mkdir -p data models && PYTHONPATH=. python3 utils/get_data.py"
    )

    base_rate_preprocess = BashOperator(
        task_id="base_rate_preprocess",
        bash_command=f"cd {base_rate_dir} && mkdir -p data models && PYTHONPATH=. python3 utils/preprocess.py"
    )

    base_rate_train = BashOperator(
        task_id="base_rate_train",
        bash_command=f"cd {base_rate_dir} && mkdir -p data models && PYTHONPATH=. python3 train.py"
    )

    base_rate_test = BashOperator(
        task_id="base_rate_test",
        bash_command=f"cd {base_rate_dir} && mkdir -p data models && PYTHONPATH=. python3 test.py"
    )

    base_rate_explain = BashOperator(
        task_id="base_rate_explain",
        bash_command=f"cd {base_rate_dir} && mkdir -p data models && PYTHONPATH=. python3 explain.py"
    )

    base_rate_interpret = BashOperator(
        task_id="base_rate_interpret",
        bash_command=f"cd {base_rate_dir} && mkdir -p data models && PYTHONPATH=. python3 interpret_xai.py"
    )

    base_rate_get_data >> base_rate_preprocess >> base_rate_train >> base_rate_test >> base_rate_explain >> base_rate_interpret

    # =========================================================================
    # 3. REAL ESTATE MODEL PIPELINE (부동산 예측 모델)
    # =========================================================================
    real_estate_dir = "/opt/airflow/dags/scripts/ml/real_estate"

    real_estate_get_data = BashOperator(
        task_id="real_estate_get_data",
        bash_command=f"cd {real_estate_dir} && mkdir -p data models && PYTHONPATH=. python3 utils/get_data.py"
    )

    real_estate_preprocess = BashOperator(
        task_id="real_estate_preprocess",
        bash_command=f"cd {real_estate_dir} && mkdir -p data models && PYTHONPATH=. python3 utils/preprocess.py"
    )

    real_estate_train = BashOperator(
        task_id="real_estate_train",
        bash_command=f"cd {real_estate_dir} && mkdir -p data models && PYTHONPATH=. python3 train.py"
    )

    real_estate_test = BashOperator(
        task_id="real_estate_test",
        bash_command=f"cd {real_estate_dir} && mkdir -p data models && PYTHONPATH=. python3 test.py"
    )

    real_estate_explain = BashOperator(
        task_id="real_estate_explain",
        bash_command=f"cd {real_estate_dir} && mkdir -p data models && PYTHONPATH=. python3 explain.py"
    )

    real_estate_interpret = BashOperator(
        task_id="real_estate_interpret",
        bash_command=f"cd {real_estate_dir} && mkdir -p data models && PYTHONPATH=. python3 interpret_xai.py"
    )

    real_estate_get_data >> real_estate_preprocess >> real_estate_train >> real_estate_test >> real_estate_explain >> real_estate_interpret


