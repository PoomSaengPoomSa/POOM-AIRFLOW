# 🔄 POOM-AIRFLOW

> **POOM** — PB(Private Banker) 업무 지원 AI Assistant 플랫폼의 MLOps 데이터 파이프라인

---

## 📌 개요

POOM 플랫폼의 Apache Airflow 기반 MLOps 파이프라인 레포지토리입니다.
경제지표 데이터를 주기적으로 수집·전처리하고, 예측 모델 학습 및 배포를 자동화합니다.
AWS S3를 공유 스토리지로 활용하며 `poom-network`를 통해 다른 서비스와 연동됩니다.

---

## 🗂 프로젝트 구조
POOM-AIRFLOW/

├── dags/                       # Airflow DAG 정의

├── config/                     # Airflow 설정 파일 (airflow.cfg)

├── data/                       # 로컬 데이터 저장소

├── logs/dag_processor/         # DAG 처리 로그

├── .github/workflows/          # GitHub Actions CI/CD

├── Dockerfile

├── docker-compose-airflow.yml  # Airflow 전체 서비스 구성

└── README.md

---

## ⚙️ 기술 스택

| 분류 | 기술 |
|---|---|
| **워크플로우** | Apache Airflow (CeleryExecutor) |
| **메시지 브로커** | Redis 7.2 |
| **메타데이터 DB** | PostgreSQL 13 |
| **스토리지** | AWS S3 (rclone FUSE 마운트) |
| **모니터링** | Celery Flower (port 5555) |
| **인프라** | Docker Compose, GitHub Actions, `poom-network` |

---

## 🧩 서비스 구성

`docker-compose-airflow.yml` 기준으로 아래 서비스가 실행됩니다.

| 서비스 | 역할 | 포트 |
|---|---|---|
| `airflow-apiserver` | Airflow API 서버 | 8080 |
| `airflow-scheduler` | DAG 스케줄링 | - |
| `airflow-worker` | Celery 작업 실행 | - |
| `airflow-dag-processor` | DAG 파일 파싱 | - |
| `airflow-triggerer` | 비동기 트리거 처리 | - |
| `postgres` | Airflow 메타데이터 저장 | - |
| `redis` | Celery 브로커 | 6379 |
| `s3-mounter` | S3 버킷 FUSE 마운트 (rclone) | - |
| `flower` | Celery 워커 모니터링 | 5555 |

---

## 🚀 실행 방법

```bash
# 1. 환경변수 설정
cp .env.example .env

# 2. AIRFLOW_UID 설정 (Linux)
echo -e "AIRFLOW_UID=$(id -u)" >> .env

# 3. 서비스 실행
docker compose -f docker-compose-airflow.yml up -d

# 4. Airflow UI 접속
# http://localhost:8080  (기본 계정: airflow / airflow)

# 5. Celery Flower 모니터링
# http://localhost:5555
```

> **최소 사양:** RAM 4GB 이상, CPU 2코어 이상, 디스크 10GB 이상

---

## 🔗 연관 레포지토리

| 레포 | 역할 |
|---|---|
| [POOM-BACK](https://github.com/PoomSaengPoomSa/POOM-BACK) | FastAPI 백엔드 서버 |
| [POOM-AI](https://github.com/PoomSaengPoomSa/POOM-AI) | LangGraph 멀티 에이전트 |
| [POOM-MLFLOW](https://github.com/PoomSaengPoomSa/POOM-MLFLOW) | 모델 실험 관리 |
| [POOM-ELK](https://github.com/PoomSaengPoomSa/POOM-ELK) | 로그 모니터링 |

---

> 우리FISA AI 엔지니어링 1팀 | POOM 프로젝트
