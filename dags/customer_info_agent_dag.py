from datetime import datetime, timedelta
import pendulum
from airflow import DAG  # type: ignore
from airflow.operators.bash import BashOperator  # type: ignore

# 한국 시간대(KST) 정의
kst = pendulum.timezone("Asia/Seoul")

# 1. 기본 설정 (Default Arguments)
default_args = {
    "owner": "poom_ai_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 5, 28, tzinfo=kst),  # KST 기준 날짜
    "email": ["admin@poom.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,                           # 일시적 API/네트워크 에러 대비 2회 자동 재시도
    "retry_delay": timedelta(minutes=5),    # 재시도 전 5분 대기
}

# 2. DAG 정의
with DAG(
    "customer_info_agent_daily_pipeline",
    default_args=default_args,
    description="VIP Customer Asset & Churn Risk Analysis Daily Agent",
    schedule="0 8 * * *",                   # 매일 아침 8시 정각에 자동 기동 (KST 기준)
    catchup=False,                          # 활성화 시 과거 날짜들의 백필이 자동 수행되지 않도록 방지
    tags=["poom", "ai_agent", "customer", "langgraph"],
) as dag:

    # [TASK 1] 전날 CRM 데이터 동기화 완료 대기 및 검증 (가상 시뮬레이션)
    wait_for_crm_sync = BashOperator(
        task_id="wait_for_crm_sync",
        bash_command="echo '=== [STEP 1] CRM 고객 데이터베이스 동기화 완료 상태를 검증합니다 ==='",
    )

    # [TASK 2] Customer Info Agent 전체 고객 실행 (핵심)
    run_customer_info_agent = BashOperator(
        task_id="run_customer_info_agent",
        bash_command="""
        cd /opt/airflow/data/scripts && python3 -m agent.customer.run_info
        """,
        # 윈도우 로컬 환경에서 테스트할 경우 아래의 윈도우 절대 경로로 치환하여 사용합니다:
        # bash_command="cd c:\\Users\\jongh\\Working_Directory\\poom\\POOM-AIRFLOW\\data\\scripts && c:\\Users\\jongh\\Working_Directory\\poom\\POOM-AI\\.venv\\Scripts\\python.exe -m agent.customer.run_info"
    )

    # [TASK 3] 에이전트 완료 후 알림 전송 (가상 시뮬레이션)
    send_daily_summary = BashOperator(
        task_id="send_daily_summary",
        bash_command="echo '=== [STEP 3] 전체 고객 리밸런싱 및 이탈 분석 완료 인앱 알림을 발송합니다 ==='",
    )

    # 3. 태스크 실행 순서 (의존성 연결)
    wait_for_crm_sync >> run_customer_info_agent >> send_daily_summary