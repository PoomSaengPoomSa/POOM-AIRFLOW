FROM apache/airflow:3.0.6

USER root
RUN apt-get update && apt-get install -y libgomp1 && apt-get clean

USER airflow
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.1.4 \
    deep-translator \
    python-dotenv \
    requests \
    mlflow \
    optuna \
    scikit-learn \
    "optuna-integration[mlflow]" \
    feedparser \
    elasticsearch \
    boto3 \
    psycopg2-binary \
    pymysql \
    joblib \
    lightgbm \
    catboost \
    xgboost==2.1.4 \
    langchain \
    langchain-openai \
    langgraph \
    pydantic-settings \
    sqlalchemy \
    shap==0.47.1 \
    matplotlib \
    cryptography==42.0.8 \
    cffi==1.17.1
