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

    # 1. 금값 학습 태스크 (S3 동기화 dags/scripts 경로 사용)
    train_gold = BashOperator(
        task_id="train_gold_model",
        bash_command="python3 /opt/airflow/dags/scripts/ml/run_gold.py",
        # 윈도우 로컬 환경 테스트용 경로 및 가상환경 명시:
        # bash_command="c:\\Users\\yu\\Documents\\Woorifisa\\poom\\POOM-AI\\.venv\\Scripts\\python.exe c:\\Users\\yu\\Documents\\Woorifisa\\poom\\POOM-AIRFLOW\\data\\scripts\\ml\\run_gold.py"
    )

    # 2. 부동산 학습 태스크 (S3 동기화 dags/scripts 경로 사용)
    train_real_estate = BashOperator(
        task_id="train_real_estate_model",
        bash_command="python3 /opt/airflow/dags/scripts/ml/run_real_estate.py",
        # 윈도우 로컬 환경 테스트용 경로 및 가상환경 명시:
        # bash_command="c:\\Users\\yu\\Documents\\Woorifisa\\poom\\POOM-AI\\.venv\\Scripts\\python.exe c:\\Users\\yu\\Documents\\Woorifisa\\poom\\POOM-AIRFLOW\\data\\scripts\\ml\\run_real_estate.py"
    )

    # 3. 기준금리 학습 태스크 (S3 동기화 dags/scripts 경로 사용)
    train_base_rate = BashOperator(
        task_id="train_base_rate_model",
        bash_command="python3 /opt/airflow/dags/scripts/ml/run_base_rate.py",
        # 윈도우 로컬 환경 테스트용 경로 및 가상환경 명시:
        # bash_command="c:\\Users\\yu\\Documents\\Woorifisa\\poom\\POOM-AI\\.venv\\Scripts\\python.exe c:\\Users\\yu\\Documents\\Woorifisa\\poom\\POOM-AIRFLOW\\data\\scripts\\ml\\run_base_rate.py"
    )

    # 태스크 의존성을 지정하지 않음으로써 3개 모델이 완벽한 병렬(Parallel)로 동작합니다.
