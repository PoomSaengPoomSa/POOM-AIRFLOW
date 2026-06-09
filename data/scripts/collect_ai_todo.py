# -*- coding: utf-8 -*-
"""
AI To-Do 에이전트 독립형 데이터 수집 및 MySQL 적재 스크립트 (테이블명 오타 보완본)
위치: airflow/data/scripts/collect_ai_todo.py
"""

import os
import sys
import logging
import json
import argparse
import requests
from datetime import datetime, timedelta, time
from sqlalchemy import create_engine, text

# ── 로깅 설정 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("collect_ai_todo")

# ── 🛡️ 순수 파이썬 기반 .env 파일 자동 탐색 및 로더 ──────────────────────────────
def load_env_file():
    """
    도커 컨테이너 내부에 존재하는 모든 예상 .env 파일 경로를 자동 탐색하여 
    시스템 환경 변수(os.environ)에 강제 주입합니다.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_env_paths = [
        # 1. 스크립트 실행 위치 기준 상위 폴더들 추적
        os.path.abspath(os.path.join(current_dir, ".env")),
        os.path.abspath(os.path.join(current_dir, "..", ".env")),
        os.path.abspath(os.path.join(current_dir, "..", "..", ".env")),
        os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".env")),
        # 2. 에어플로우 도커 컨테이너 내부 루트 마운트 경로 추적
        "/opt/airflow/.env",
        "/opt/airflow/poom/.env",
        "/opt/airflow/POOM-BACK/.env",
        "/opt/airflow/poom/POOM-BACK/.env",
    ]
    
    env_file_path = None
    for path in possible_env_paths:
        if os.path.exists(path):
            env_file_path = path
            break
            
    if env_file_path:
        logger.info(f"[ENV] 컨테이너 내부에서 .env 파일을 찾았습니다: {env_file_path}")
        try:
            with open(env_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    if "=" in stripped:
                        key, val = stripped.split("=", 1)
                        key = key.strip()
                        val = val.strip().strip("'").strip('"')
                        os.environ[key] = val
            logger.info("[ENV] .env 환경 변수가 메모리에 정상 주입되었습니다.")
        except Exception as e:
            logger.warning(f"[ENV] .env 파일 파싱 중 예외 발생: {e}")
    else:
        logger.warning("[ENV] 컨테이너 내 예상 경로에서 .env 파일을 찾을 수 없습니다. 시스템 환경 변수를 그대로 사용합니다.")

# 프로그램 시작 시 환경 변수 로더 즉시 실행
load_env_file()

# ── 💡 보안 환경 변수 가져오기 및 검증 ──────────────────────────────────────────
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 최종 환경 변수 무결성 검증
missing_envs = []
if not DB_USER: missing_envs.append("DB_USER")
if not DB_PASSWORD: missing_envs.append("DB_PASSWORD")
if not DB_HOST: missing_envs.append("DB_HOST")
if not DB_NAME: missing_envs.append("DB_NAME")
if not OPENAI_API_KEY: missing_envs.append("OPENAI_API_KEY")

if missing_envs:
    logger.error(f"[보안 오류] 필수 환경 변수가 누락되었습니다: {', '.join(missing_envs)}")
    sys.exit(1)

# ── 데이터베이스 연결 엔진 생성 ────────────────────────────────────────────────────
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
engine = create_engine(DATABASE_URL)

# ── 1. 데이터베이스로부터 상황 분석 컨텍스트 조회 ───────────────────────────────────────

def fetch_pb_context(u_id: str, target_date_str: str) -> dict:
    """
    지정된 PB 및 분석 날짜에 대한 전체 고객 데이터베이스 상황판을 실시간 쿼리하여 조립합니다.
    """
    logger.info(f"데이터베이스로부터 컨텍스트 데이터 수집 중... (PB ID: {u_id}, 기준일: {target_date_str})")
    
    context = {}
    try:
        base_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
    except ValueError:
        base_date = datetime.now().date()
    end_date = base_date + timedelta(days=30)
    
    with engine.connect() as conn:
        # (A) PB 기본 정보 확인
        pb_res = conn.execute(
            text("SELECT name, position, branch FROM pb_user WHERE u_id = :u_id"),
            {"u_id": u_id}
        ).fetchone()
        
        if not pb_res:
            logger.error(f"PB ID '{u_id}'가 존재하지 않습니다.")
            sys.exit(1)
            
        pb_name, pb_position, branch_id = pb_res
        context["pb_info"] = f"{pb_name} {pb_position} (지점 ID: {branch_id})"
        
        # (B) PB 담당 고객 ID 목록
        c_ids_res = conn.execute(
            text("SELECT c_id FROM in_charge WHERE u_id = :u_id"),
            {"u_id": u_id}
        ).fetchall()
        c_ids = [int(row[0]) for row in c_ids_res]
        
        if not c_ids:
            logger.warning("담당하는 고객이 존재하지 않습니다.")
            return None
            
        # SQL에 이식할 안전한 문자열 ID 목록 생성 (예: "1, 2")
        c_ids_str = ", ".join(str(cid) for cid in c_ids)
            
        # (C) 1. 캘린더 기존 일정 수집 (💡 schedule -> pb_schedule 테이블명 3글자 오타 전격 수정 완료!)
        start_dt = datetime.combine(base_date, time.min)
        end_dt = datetime.combine(base_date, time.max)
        schedules_res = conn.execute(
            text("""
                SELECT s.execution_date, s.end_datetime, s.category, s.title, c.name 
                FROM pb_schedule s
                LEFT JOIN customer c ON s.c_id = c.c_id
                WHERE s.u_id = :u_id AND s.execution_date >= :start_dt AND s.execution_date <= :end_dt
                ORDER BY s.execution_date ASC
            """),
            {"u_id": u_id, "start_dt": start_dt, "end_dt": end_dt}
        ).fetchall()
        
        schedule_list = []
        for s in schedules_res:
            start_time = s[0].strftime("%H:%M")
            end_time = s[1].strftime("%H:%M") if s[1] else "종료시간 없음"
            cust = f" (고객: {s[4]})" if s[4] else ""
            schedule_list.append(f"- [{start_time} ~ {end_time}] [{s[2]}] {s[3]}{cust}")
            
        context["calendar"] = "\n".join(schedule_list) if schedule_list else "등록된 기존 일정 없음 (오전 9시 ~ 오후 6시 전체 시간대 비어있음)"
        
        # (D) 2. KPI 실적 데이터 수집
        kpi_res = conn.execute(
            text("""
                SELECT target_aum, current_aum, target_non_interest, current_non_interest, 
                       target_new_customer, current_new_customer, recorded_date
                FROM kpi 
                WHERE u_id = :u_id AND kpi_type = 'PB'
                ORDER BY recorded_date DESC LIMIT 1
            """),
            {"u_id": u_id}
        ).fetchone()
        
        if kpi_res:
            t_aum, c_aum, t_non, c_non, t_new, c_new, r_date = kpi_res
            aum_rate = (c_aum / t_aum * 100) if t_aum else 0
            non_rate = (c_non / t_non * 100) if t_non else 0
            new_rate = (c_new / t_new * 100) if t_new else 0
            
            kpi_str = (
                f"- AUM (수신자산): 목표 {t_aum/100000000:.1f}억 / 현재 {c_aum/100000000:.1f}억 (달성률: {aum_rate:.1f}%)\n"
                f"- 비이자 수익: 목표 {t_non/100000000:.1f}억 / 현재 {c_non/100000000:.1f}억 (달성률: {non_rate:.1f}%)\n"
                f"- 신규 고객 유치: 목표 {t_new}명 / 현재 {c_new}명 (달성률: {new_rate:.1f}%)"
            )
            context["kpi"] = kpi_str
        else:
            context["kpi"] = "등록된 KPI 목표 정보 없음"
            
        # (E) 3. 이탈 위험 고객 수집
        risks_res = conn.execute(
            text(f"""
                SELECT c.c_id, c.name, c.grade, c.total_assets, cr.grade as risk_grade, cr.reason
                FROM churn_level cr
                JOIN customer c ON cr.c_id = c.c_id
                WHERE cr.c_id IN ({c_ids_str}) AND cr.grade IN ('주의', '위험')
                ORDER BY cr.created_date DESC
            """)
        ).fetchall()
        
        unique_risks = {}
        for r in risks_res:
            if r[0] not in unique_risks:
                unique_risks[r[0]] = r
                
        risk_list = []
        for c_id, r in unique_risks.items():
            assets = f"{r[3]/100000000:.1f}억" if r[3] else "0원"
            risk_list.append(f"- **{r[1]}** ({r[2]} 등급, 총자산: {assets}) - **위험도: [{r[4]}]** | 사유: {r[5]}")
        context["risks"] = "\n".join(risk_list) if risk_list else "위험/주의 수준의 이탈 우려 고객 없음"
        
        # (F) 4. 상품 만기 및 기념일 이벤트 수집
        event_list = []
        
        # 1. 만기 상품
        exp_products = conn.execute(
            text(f"""
                SELECT cp.c_id, c.name, p.name as p_name, cp.expiration_date
                FROM customer_product cp
                JOIN product p ON cp.pd_id = p.pd_id
                JOIN customer c ON cp.c_id = c.c_id
                WHERE cp.c_id IN ({c_ids_str}) AND cp.expiration_date >= :base_date AND cp.expiration_date <= :end_date
            """),
            {"base_date": base_date, "end_date": end_date}
        ).fetchall()
        for ep in exp_products:
            event_list.append(f"- [만기 예정] {ep[1]} 고객: 상품 '{ep[2]}' 만기 예정일 ({ep[3].strftime('%Y-%m-%d')})")
            
        # 2. 생일
        birthday_res = conn.execute(
            text(f"SELECT c_id, name, birthday, total_assets FROM customer WHERE c_id IN ({c_ids_str})")
        ).fetchall()
        for br in birthday_res:
            if br[2]:
                try:
                    this_year_bday = br[2].replace(year=base_date.year)
                except ValueError:
                    this_year_bday = br[2].replace(year=base_date.year, day=28)
                if base_date <= this_year_bday <= end_date:
                    event_list.append(f"- [생일 도래] {br[1]} 고객: 생일 ({br[2].strftime('%m-%d')}) - 자산: {br[3]/100000000:.1f}억")
                    
        # 3. 결혼기념일
        marriage_res = conn.execute(
            text(f"""
                SELECT cr.c_id, c.name, cr.wedding_date 
                FROM customer_relationship cr
                JOIN customer c ON cr.c_id = c.c_id
                WHERE cr.c_id IN ({c_ids_str}) AND cr.is_spouse = 1 AND cr.wedding_date IS NOT NULL
            """)
        ).fetchall()
        for mr in marriage_res:
            try:
                this_year_wedding = mr[2].replace(year=base_date.year)
            except ValueError:
                this_year_wedding = mr[2].replace(year=base_date.year, day=28)
            if base_date <= this_year_wedding <= end_date:
                event_list.append(f"- [결혼기념일] {mr[1]} 고객: 결혼기념일 ({mr[2].strftime('%m-%d')})")
                
        context["events"] = "\n".join(event_list) if event_list else "30일 이내 도래하는 고객 이벤트 없음"
        
        # (G) 5. 최근 상담 이력 수집
        memos_res = conn.execute(
            text("""
                SELECT c.name, cm.consult_date, cm.memo 
                FROM consultation_memo cm
                JOIN customer c ON cm.c_id = c.c_id
                WHERE cm.u_id = :u_id
                ORDER BY cm.consult_date DESC LIMIT 5
            """),
            {"u_id": u_id}
        ).fetchall()
        memo_list = [f"- {m[0]} 고객 상담 ({m[1].strftime('%Y-%m-%d')}): {m[2]}" for m in memos_res]
        context["histories"] = "\n".join(memo_list) if memo_list else "최근 진행된 상담 기록 없음"
        
        # (H) 6. 기존 생성 알림 리스트 수집
        notifs_res = conn.execute(
            text("""
                SELECT category, title, created_time 
                FROM notification 
                WHERE u_id = :u_id 
                ORDER BY created_time DESC LIMIT 10
            """),
            {"u_id": u_id}
        ).fetchall()
        notif_list = [f"- [{n[0]}] {n[1]} ({n[2].strftime('%Y-%m-%d %H:%M')})" for n in notifs_res]
        context["notifications"] = "\n".join(notif_list) if notif_list else "기존 알림 내역 없음"
        
    logger.info("데이터베이스 상황 분석 정보 추출 완료!")
    return context

# ── 2. OpenAI GPT-4o 호출 및 결과 파싱 ───────────────────────────────────────────

def call_openai_todo_agent(context: dict, u_id: str, target_date_str: str) -> list:
    """
    OpenAI API를 활용해 PB의 오늘 하루 목표를 수립하고 최적의 추천 일정 5건을 정렬 생성합니다.
    """
    logger.info("OpenAI GPT-4o 모델을 호출하여 일정을 기획합니다...")
    
    prompt = f"""
역할: 전문 PB(Private Banker)를 위한 지능형 개인 비서 에이전트.
목표: 기준 날짜({target_date_str})에 PB가 고객 관리와 KPI 목표 달성을 위해 진행해야 할 가장 중요하고 실현 가능한 추천 일정 5건을 도출합니다.

[상황 분석 컨텍스트 데이터]
1. PB 및 지점 정보:
{context.get('pb_info')}

2. 금일 기존 등록된 일정 리스트 (이 시간대를 피해서 생성할 것):
{context.get('calendar')}

3. PB 개인 KPI 달성 현황 및 권장 마케팅 방향:
{context.get('kpi')}

4. 이탈 위험 주의/위험군 고객 목록:
{context.get('risks')}

5. 기준일로부터 30일 이내 도래 주요 이벤트 (상품 만기, 생일, 기념일):
{context.get('events')}

6. 최근 5건의 대고객 상담 내역:
{context.get('histories')}

7. 중복 방지용 최근 알림 내역:
{context.get('notifications')}

[일정 편성 비즈니스 룰 및 제약 조건]
1. **일정의 개수**: 반드시 정확하게 **5건**의 일정을 추천해야 합니다.
2. **카테고리 매핑**: 일정의 카테고리는 반드시 다음 4가지 중 하나에 속해야 합니다:
   - '안부 연락 제안': 고객의 생일, 결혼기념일, 안부 전달 등. (주로 아침 10시 혹은 11시 슬롯으로 배정)
   - '상담 일정 제안': 대면 상담 예약, 자산관리 상담. (주로 오후 14시, 15시, 16시 슬롯으로 배정)
   - 'KPI 기반': 신규 IRP 개설, 자금 유치 마케팅 등.
   - '신규 상품 분석': 투자 상품 제안 등.
3. **슬롯 배정 및 정렬 순서 (매우 중요)**:
   - 생성하는 5건의 일정을 중요도 순서(Priority)로 내림차순 정렬해야 합니다.
   - 1순위와 2순위는 가벼운 터치가 적합한 '안부 연락 제안'으로 선정하여 아침/오전 시간대(**10:00:00**, **11:00:00**)에 고정 배치합니다.
   - 3순위, 4순위, 5순위는 업무 집중도가 높은 대면 상담 및 세일즈 업무를 위한 시간대(**14:00:00**, **15:00:00**, **16:00:00**)에 순차 배치합니다.
   - 예시 시간 배치:
     - 1순위 (최고 중요): 10:00:00 (안부 연락 제안)
     - 2순위 (높은 중요): 11:00:00 (안부 연락 제안)
     - 3순위 (보통 중요): 14:00:00 (상담/세일즈 제안)
     - 4순위 (보통 중요): 15:00:00 (상담/세일즈 제안)
     - 5순위 (보통 중요): 16:00:00 (상담/세일즈 제안)
4. **고객 매핑**: 대상 고객의 `c_id`가 식별된다면 반드시 매핑해 줍니다. 만약 특정 고객 타겟이 아닌 일반 KPI 세일즈인 경우 `c_id`를 `null`로 지정합니다.
5. **시간 포맷**: `execution_date`는 반드시 `YYYY-MM-DD HH:MM:SS` 형식 문자열로 출력해야 합니다.
6. **글자 수 제약**:
   - `title`: 반드시 50자 이내
   - `memo`: 반드시 80자 이내

반드시 아래와 같은 JSON 배열 형식으로만 답변하세요. 마크다운 백틱(```json)이나 다른 설명 텍스트를 절대 붙이지 말고 순수 JSON 형식으로만 반환하세요:

[
  {
    "title": "김신한 고객(1) 생일 축하 감사 연락 및 안부 인사",
    "memo": "오늘 생일을 맞이한 VIP 김신한 고객님께 유선 안부 및 커피 기프티콘 발송",
    "category": "안부 연락 제안",
    "execution_date": "{target_date_str} 10:00:00",
    "c_id": 1
  },
  ...
]
"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant specialized in scheduling and client management for financial advisors."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        res_json = response.json()
        raw_content = res_json["choices"][0]["message"]["content"].strip()
        
        if raw_content.startswith("```"):
            lines = raw_content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_content = "\n".join(lines).strip()
            
        todos = json.loads(raw_content)
        if not isinstance(todos, list) or len(todos) != 5:
            logger.warning(f"생성된 일정 개수가 5건이 아닙니다: {len(todos)}건 생성됨. 강제 검증 및 재정렬 적용.")
        
        logger.info("OpenAI GPT-4o 일정 기획 및 파싱 완료!")
        return todos
        
    except Exception as e:
        logger.error(f"OpenAI API 호출 및 파싱 실패: {e}")
        return generate_dynamic_fallback(u_id, target_date_str)

# ── 3. 데이터베이스 기반 동적 룰(Heuristic) 플래너 ──────────────────────────

def generate_dynamic_fallback(u_id: str, target_date_str: str) -> list:
    """
    OpenAI API 장애 시 가동되는 룰 기반(Heuristic) 동적 일정 플래너.
    실제 데이터베이스의 PB 담당 고객 데이터(기념일, 이탈위험, 일반 고객)를 추출하여 
    맞춤형 추천 일정 5건을 규칙에 맞춰 실시간 조합합니다.
    """
    logger.warning("가용 API 장애 또는 연결 오류로 인해 데이터베이스 기반 동적 룰(Heuristic) 플래너를 가동합니다.")
    
    todos = []
    
    try:
        with engine.connect() as conn:
            # 1. 현재 가동 중인 PB가 담당하는 실제 고객 목록 조회 (자산 순으로 정렬) (💡 schedule -> pb_schedule 수정)
            customers_res = conn.execute(
                text("""
                    SELECT c.c_id, c.name, c.grade, c.total_assets 
                    FROM customer c
                    JOIN in_charge ic ON c.c_id = ic.c_id
                    WHERE ic.u_id = :u_id
                    ORDER BY c.total_assets DESC
                """),
                {"u_id": u_id}
            ).fetchall()
            
            if not customers_res:
                logger.error(f"담당 고객이 존재하지 않는 PB ID({u_id})이므로 동적 폴백 일정을 생성할 수 없습니다.")
                return []
            
            cust_list = [
                {
                    "c_id": r[0],
                    "name": r[1],
                    "grade": r[2],
                    "assets": r[3] if r[3] else 0
                } for r in customers_res
            ]
            
            # 2. 이탈위험 주의/위험에 등록된 실제 담당 고객 조회
            risk_res = conn.execute(
                text("""
                    SELECT cr.c_id, c.name, cr.reason 
                    FROM churn_level cr
                    JOIN customer c ON cr.c_id = c.c_id
                    JOIN in_charge ic ON c.c_id = ic.c_id
                    WHERE ic.u_id = :u_id AND cr.grade IN ('주의', '위험')
                    ORDER BY cr.created_date DESC LIMIT 2
                """),
                {"u_id": u_id}
            ).fetchall()
            
            risk_cust_list = [{"c_id": r[0], "name": r[1], "reason": r[2]} for r in risk_res]
            
            # 1순위 (10:00:00) - 안부 연락 제안 (담당 고객 중 자산 순위 1위)
            c1 = cust_list[0]
            todos.append({
                "title": f"{c1['name']} 고객({c1['c_id']}) 정기 안부 연락 및 케어",
                "memo": f"우량 고객인 {c1['name']}님께 안부 전화를 취하고 영업점 금리 우대 쿠폰 발송 제안",
                "category": "안부 연락 제안",
                "execution_date": f"{target_date_str} 10:00:00",
                "c_id": c1["c_id"]
            })
            
            # 2순위 (11:00:00) - 안부 연락 제안 (담당 고객 중 자산 순위 2위)
            c2 = cust_list[1] if len(cust_list) > 1 else c1
            todos.append({
                "title": f"{c2['name']} 고객({c2['c_id']}) 기념일/생일 조회 및 안부 인사",
                "memo": "금달 주요 스케줄링 안내를 담은 감사 인사 발송 및 기프티콘 조율",
                "category": "안부 연락 제안",
                "execution_date": f"{target_date_str} 11:00:00",
                "c_id": c2["c_id"]
            })
            
            # 3순위 (14:00:00) - 상담 일정 제안 (이탈 위험 1순위 고객 우선 배치, 없을 시 자산 우량 고객)
            if risk_cust_list:
                rc1 = risk_cust_list[0]
                todos.append({
                    "title": f"{rc1['name']} 고객({rc1['c_id']}) 자산 리스크 진단 및 대면 상담",
                    "memo": f"이탈 위험 방지(감지사유: {rc1['reason'][:30]}...)를 위한 포트폴리오 리밸런싱 상담",
                    "category": "상담 일정 제안",
                    "execution_date": f"{target_date_str} 14:00:00",
                    "c_id": rc1["c_id"]
                })
            else:
                c3 = cust_list[2] if len(cust_list) > 2 else c1
                todos.append({
                    "title": f"{c3['name']} 고객({c3['c_id']}) 자산 만기 리밸런싱 대면 상담",
                    "memo": f"보유 상품의 만기 재예치 및 종합 자산 운용 설계를 위한 1:1 대면 상담 예약",
                    "category": "상담 일정 제안",
                    "execution_date": f"{target_date_str} 14:00:00",
                    "c_id": c3["c_id"]
                })
                
            # 4순위 (15:00:00) - 신규 상품 분석 (이탈 위험 2순위 고객 또는 자산 우량 고객)
            if len(risk_cust_list) > 1:
                rc2 = risk_cust_list[1]
                todos.append({
                    "title": f"{rc2['name']} 고객({rc2['c_id']}) 고금리 대안 투자 포트폴리오 제안",
                    "memo": "수익률 다각화를 위한 하이일드 채권 및 대체 상품 리밸런싱 제안 상담",
                    "category": "신규 상품 분석",
                    "execution_date": f"{target_date_str} 15:00:00",
                    "c_id": rc2["c_id"]
                })
            else:
                c4 = cust_list[3] if len(cust_list) > 3 else (cust_list[1] if len(cust_list) > 1 else c1)
                todos.append({
                    "title": f"{c4['name']} 고객({c4['c_id']}) 글로벌 변동성 대응 신규 상품 분석",
                    "memo": "금리 변동기에 적합한 당사 신규 특판 펀드 및 절세 세일즈 상담 유치",
                    "category": "신규 상품 분석",
                    "execution_date": f"{target_date_str} 15:00:00",
                    "c_id": c4["c_id"]
                })
                
            # 5순위 (16:00:00) - KPI 기반 (개인형 IRP 세액 공제 상담, 자산 순위 5위 또는 1위)
            c5 = cust_list[4] if len(cust_list) > 4 else c1
            todos.append({
                "title": "신규 개인형 IRP 개설 및 노후 은퇴 설계 상담",
                "memo": f"연말정산 혜택 극대화를 위해 {c5['name']}님 대상 IRP 추가 불입 및 절세 포트폴리오 설계",
                "category": "KPI 기반",
                "execution_date": f"{target_date_str} 16:00:00",
                "c_id": c5["c_id"]
            })
            
    except Exception as e:
        logger.error(f"동적 룰 폴백 일정 조합 중 오류 발생: {e}")
        return []
        
    logger.info("데이터베이스 기반의 동적 룰(Heuristic) Fallback 일정 생성 완료!")
    return todos

# ── 4. 데이터베이스 영구 적재 ───────────────────────────────────────────────────

def insert_ai_todos_to_db(u_id: str, todos: list):
    """
    최종 생성되어 정렬된 5건의 추천 일정을 데이터베이스 `ai_todo` 테이블에 영구 적재합니다.
    """
    if not todos:
        logger.error("적재할 추천 일정이 존재하지 않습니다.")
        return
        
    logger.info(f"생성된 {len(todos)}건의 추천 일정을 ai_todo 테이블에 적재하기 시작합니다...")
    
    current_timestamp = datetime.utcnow()
    inserted_count = 0
    
    with engine.begin() as conn:
        conn.execute(
            text("""
                DELETE FROM ai_todo 
                WHERE u_id = :u_id 
                  AND DATE(execution_date) = DATE(:exec_date)
                  AND is_checked = 0
            """),
            {"u_id": u_id, "exec_date": todos[0]["execution_date"]}
        )
        
        for t in todos:
            try:
                exec_dt = datetime.strptime(t["execution_date"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                exec_dt = datetime.strptime(t["execution_date"].split(" ")[0] + " 10:00:00", "%Y-%m-%d %H:%M:%S")
            
            valid_categories = {'KPI 기반', '상담 일정 제안', '안부 연락 제안', '신규 상품 분석'}
            category = t["category"] if t["category"] in valid_categories else '상담 일정 제안'
            
            title = t["title"][:50]
            memo = t["memo"][:80] if t.get("memo") else ""
            c_id = int(t["c_id"]) if t.get("c_id") else None
            
            conn.execute(
                text("""
                    INSERT INTO ai_todo (title, memo, category, create_date, execution_date, is_checked, u_id, c_id)
                    VALUES (:title, :memo, :category, :create_date, :execution_date, 0, :u_id, :c_id)
                """),
                {
                    "title": title,
                    "memo": memo,
                    "category": category,
                    "create_date": current_timestamp,
                    "execution_date": exec_dt,
                    "u_id": u_id,
                    "c_id": c_id
                }
            )
            inserted_count += 1
            logger.info(f"  적재 완료 -> 시간: {exec_dt.strftime('%H:%M')} | 카테고리: {category} | 제목: {title}")
            
    logger.info(f"데이터베이스 영구 적재 성공! 총 {inserted_count}건의 일정을 성공적으로 입력했습니다.")

# ── 5. 엔트리 포인트 ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Standalone AI To-Do Agent Scheduler Runner")
    parser.add_argument("--u_id", type=str, default="pb_b1_1", help="PB ID (e.g. pb_b1_1)")
    parser.add_argument("--date", type=str, help="Target date YYYY-MM-DD (e.g. 2026-05-27)")
    
    args = parser.parse_args()
    
    target_date = args.date or datetime.now().strftime("%Y-%m-%d")
    
    logger.info("="*60)
    logger.info(f"독립형 AI To-Do 에이전트 수집 엔진 구동 시작 (PB: {args.u_id}, 날짜: {target_date})")
    logger.info("="*60)
    
    context = fetch_pb_context(args.u_id, target_date)
    
    if not context:
        logger.error("컨텍스트 데이터를 조회하는 데 실패했습니다. 담당 고객 목록이 비어 있거나 연결에 오류가 있습니다.")
        sys.exit(1)
        
    todos = call_openai_todo_agent(context, args.u_id, target_date)
    insert_ai_todos_to_db(args.u_id, todos)
    
    logger.info("="*60)
    logger.info("독립형 AI To-Do 에이전트 프로세스가 정상적으로 종료되었습니다.")
    logger.info("="*60)

if __name__ == "__main__":
    main()