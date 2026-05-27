from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# 1. 기본 설정 (Default Arguments)
default_args = {
    "owner": "poom_ai_team",
    "depends_on_past": False,
    "start_date": datetime(2026, 5, 26),
    "email": ["admin@poom.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,                           # 일시적 API/네트워크 에러 대비 2회 자동 재시도
    "retry_delay": timedelta(minutes=5),    # 재시도 전 5분 대기
}

# 2. DAG 정의
with DAG(
    "ai_todo_agent_daily_pipeline",
    default_args=default_args,
    description="PB AI To-Do Goal-driven Agent Daily Generation Pipeline",
    schedule="0 6 * * *",                   # Airflow 2.4+ 규격: 매일 아침 6시 정각에 자동 기동 (KST)
    catchup=False,                          # 활성화 시 과거 날짜들의 백필이 자동 수행되지 않도록 방지
    tags=["poom", "ai_agent", "langgraph"],
) as dag:

    # [TASK 1] 전날 CRM 데이터 동기화 완료 대기 및 검증 (가상 시뮬레이션)
    wait_for_crm_sync = BashOperator(
        task_id="wait_for_crm_sync",
        bash_command="echo '=== [STEP 1] CRM 고객 데이터베이스 동기화 완료 상태를 검증합니다 ==='",
    )

    # [TASK 2] AI To-Do Agent 구동 및 적재 (핵심)
    # Airflow의 ds 매크로({{ ds }})를 사용하여 해당 스케줄 실행 날짜를 동적으로 넘겨줍니다.
    # 리눅스/Docker 서버의 절대 경로에 맞춰 파이썬 실행기 및 main.py 경로를 기정의합니다.
    run_ai_todo_agent = BashOperator(
        task_id="run_ai_todo_agent",
        bash_command="""
        python3 /opt/airflow/poom/POOM-AI/agent/todo/main.py --u_id user1 --date {{ ds }}
        """,
        # 윈도우 로컬 환경에서 테스트할 경우 아래의 윈도우 절대 경로로 치환하여 사용합니다:
        # bash_command="c:\\ITStudy\\poom\\back\\.venv\\Scripts\\python.exe c:\\ITStudy\\poom\\ai\\agent\\todo\\main.py --u_id user1 --date {{ ds }}"
    )

    # [TASK 3] 에이전트 완료 후, PB 모바일/웹 알림 전송 API 트리거 (가상 시뮬레이션)
    send_daily_summary = BashOperator(
        task_id="send_daily_summary",
        bash_command="echo '=== [STEP 3] PB 모바일 앱으로 추천 일정 생성 완료 인앱 알림을 발송합니다 ==='",
    )

    # 3. 태스크 실행 순서 (의존성 연결)
    # CRM 동기화 검증이 통과해야만 에이전트가 돌고, 에이전트 성공 시 최종 알림이 전송됩니다.
    wait_for_crm_sync >> run_ai_todo_agent >> send_daily_summary
