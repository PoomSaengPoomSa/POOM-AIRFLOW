"""
SBS 뉴스 수집 → 클러스터링 파이프라인
매일 오전 6시 실행
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys

# 스크립트 경로 추가
sys.path.insert(0, "/opt/airflow/data/scripts")


def collect_news():
    from collect_news import main
    main()


def cluster_news():
    from cluster_news import main
    main()


with DAG(
    dag_id="sbs_news_pipeline",
    schedule="0 6 * * *",   # 매일 오전 6시
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["news", "elasticsearch"],
) as dag:

    collect = PythonOperator(
        task_id="collect_news",
        python_callable=collect_news,
    )

    
    collect  # 수집 완료 후 클러스터링 실행