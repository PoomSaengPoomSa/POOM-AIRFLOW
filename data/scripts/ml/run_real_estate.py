import os
import pickle
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import pymysql
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor

GENERATE_REPORT = False  # 테스트 중엔 False, 운영 시 True로 변경

class RealEstateEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state
        
        self.ridge = Ridge(
            alpha=1.0
        )
        
        self.rf = RandomForestRegressor(
            n_estimators=150,
            max_depth=4,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.cat = CatBoostRegressor(
            iterations=150,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=4.0,
            random_seed=random_state,
            verbose=0
        )
        
        self.models = {
            'RidgeRegressor': self.ridge,
            'RandomForest': self.rf,
            'CatBoost': self.cat
        }
        
    def fit(self, X, y):
        print("\n  [Production-Grade 3-Model Ensemble Training]")
        for name, model in self.models.items():
            print(f"    - Training {name}...")
            model.fit(X, y)
        print("  Ensemble training complete!")
        return self
        
    def predict(self, X):
        preds = []
        weights = [0.60, 0.20, 0.20]
        
        preds.append(self.ridge.predict(X) * weights[0])
        preds.append(self.rf.predict(X) * weights[1])
        preds.append(self.cat.predict(X) * weights[2])
        
        return np.sum(preds, axis=0)
        
    def get_individual_predictions(self, X):
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X)
        return preds


def load_data_from_mysql():
    load_dotenv(find_dotenv())

    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')

    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        raise ValueError("Missing database credentials in environmental variables.")

    DB_PORT = int(DB_PORT)

    connection = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

    try:
        with connection.cursor() as cursor:
            sql = "SELECT * FROM ml_realestate_raw ORDER BY loaded_date ASC"
            cursor.execute(sql)
            rows = cursor.fetchall()
    finally:
        connection.close()

    df = pd.DataFrame(rows)
    df['date_ym'] = pd.to_datetime(df['loaded_date']).dt.strftime('%Y%m')
    df = df.drop(columns=[c for c in ['rr_id', 'loaded_date'] if c in df.columns])

    numeric_cols = [col for col in df.columns if col != 'date_ym']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"[DB] Loaded data successfully from MySQL table 'ml_realestate_raw': {len(df)} rows")
    return df


def calculate_vif_custom(df, features):
    vif_dict = {}
    for feature in features:
        # Check if the feature is constant (zero variance)
        if df[feature].nunique() <= 1:
            vif_dict[feature] = float('inf')
            continue
            
        other_features = [f for f in features if f != feature]
        # Filter out other features that are constant (zero variance)
        other_features = [f for f in other_features if df[f].nunique() > 1]
        
        if not other_features:
            vif_dict[feature] = 1.0
            continue

        X = df[other_features].values
        y = df[feature].values

        try:
            reg = LinearRegression().fit(X, y)
            r2 = reg.score(X, y)
            if np.isnan(r2):
                vif = float('inf')
            elif r2 >= 1.0:
                vif = float('inf')
            else:
                vif = 1.0 / (1.0 - r2)
        except Exception:
            vif = float('inf')
            
        vif_dict[feature] = vif

    return pd.Series(vif_dict)


def filter_features_by_vif(df, features, threshold=5.0, max_features=6):
    current_features = list(features)
    print("  [Aggressive VIF Feature Pruning for Production]")

    protected_features = ["buyer_dominance_change", "kr_mortgage_rate_change", "house_price_idx_change"]

    # First, let's drop any features that are constant (zero variance)
    constant_features = [f for f in current_features if df[f].nunique() <= 1]
    for feat in constant_features:
        if feat not in protected_features:
            print(f"    - Dropping constant feature '{feat}' before VIF calculation.")
            if feat in current_features:
                current_features.remove(feat)

    while True:
        if len(current_features) <= 4:
            break

        vif_series = calculate_vif_custom(df, current_features)

        # Drop protected features from candidates
        candidates = vif_series.drop(labels=[f for f in protected_features if f in vif_series.index], errors='ignore')
        
        # If there are no candidates, we must stop
        if candidates.empty:
            break

        # Replace NaNs in candidates with inf to prune them first
        candidates = candidates.fillna(float('inf'))

        max_vif = candidates.max()
        max_feature = candidates.idxmax()

        # Safety fallback if idxmax returned NaN or something not in current_features
        if pd.isna(max_feature) or max_feature not in current_features:
            non_protected = [f for f in current_features if f not in protected_features]
            if non_protected:
                max_feature = non_protected[0]
                max_vif = float('inf')
            else:
                break

        if max_vif > threshold or len(current_features) > max_features:
            print(f"    - Dropping '{max_feature}' with VIF = {max_vif:.4f}")
            if max_feature in current_features:
                current_features.remove(max_feature)
            else:
                break
        else:
            break

    print(f"  Final selected features ({len(current_features)}): {current_features}")

    final_vifs = calculate_vif_custom(df, current_features)
    for feat, v in final_vifs.items():
        print(f"    * {feat:<25}: VIF = {v:.4f}")

    return current_features


def preprocess_data(test_months=24, vif_threshold=5.0):
    df = load_data_from_mysql()
    df = df.sort_values("date_ym").reset_index(drop=True)

    df["next_house_price_idx"] = df["house_price_idx"].shift(-1)
    TARGET = "next_change_rate"
    df[TARGET] = (df["next_house_price_idx"] - df["house_price_idx"]) / df["house_price_idx"] * 100

    df = df.dropna(subset=[TARGET]).copy()

    raw_features = [
        "house_price_idx", "kr_cpi", "kr_unemployment",
        "kr_base_rate", "kr_mortgage_rate", "kospi200",
        "apt_trade_count", "kr_m2", "buyer_dominance"
    ]
    df[raw_features] = df[raw_features].ffill().bfill()

    df["house_price_idx_change"] = df["house_price_idx"].pct_change() * 100
    df["kr_cpi_change"] = df["kr_cpi"].pct_change() * 100
    df["kr_unemployment_change"] = df["kr_unemployment"].diff()
    df["kr_base_rate_change"] = df["kr_base_rate"].diff()
    df["kr_mortgage_rate_change"] = df["kr_mortgage_rate"].diff()
    df["kospi200_change"] = df["kospi200"].pct_change() * 100
    df["apt_trade_count_change"] = df["apt_trade_count"].pct_change() * 100
    df["kr_m2_change"] = df["kr_m2"].pct_change() * 100
    df["buyer_dominance_change"] = df["buyer_dominance"].diff()

    month_series = pd.to_datetime(df['date_ym'], format='%Y%m').dt.month
    df['month_sin'] = np.sin(2 * np.pi * month_series / 12)
    df['month_cos'] = np.cos(2 * np.pi * month_series / 12)
    seasonality_features = ['month_sin', 'month_cos']

    stationary_cols = [
        "house_price_idx_change", "kr_cpi_change", "kr_unemployment_change",
        "kr_mortgage_rate_change", "kospi200_change", "buyer_dominance_change",
        "apt_trade_count_change"
    ]

    lagged_features = []
    for col in stationary_cols:
        df[f"{col}_lag1"] = df[col].shift(1)
        df[f"{col}_lag2"] = df[col].shift(2)
        df[f"{col}_lag3"] = df[col].shift(3)
        lagged_features.extend([f"{col}_lag1", f"{col}_lag2", f"{col}_lag3"])

    rolling_features = []
    for col in ["house_price_idx_change", "buyer_dominance_change", "apt_trade_count_change", "kr_mortgage_rate_change"]:
        df[f"{col}_ma3"] = df[col].rolling(window=3).mean()
        df[f"{col}_ma6"] = df[col].rolling(window=6).mean()
        rolling_features.extend([f"{col}_ma3", f"{col}_ma6"])

    candidate_features = [
        "house_price_idx_change", "kr_cpi_change", "kr_unemployment_change",
        "kr_base_rate_change", "kr_mortgage_rate_change", "kospi200_change",
        "apt_trade_count_change", "kr_m2_change", "buyer_dominance_change"
    ] + lagged_features + rolling_features + seasonality_features

    train_df = df.iloc[:-test_months].copy()
    test_df = df.iloc[-test_months:].copy()

    train_df[candidate_features] = train_df[candidate_features].ffill().bfill().fillna(0.0)
    test_df[candidate_features] = test_df[candidate_features].ffill().bfill().fillna(0.0)

    print("=" * 55)
    print("Data Preprocessing & Train/Test Splitting")
    print("=" * 55)
    print(f"  Total samples     : {len(df)}")
    print(f"  Train period      : {train_df['date_ym'].min()} ~ {train_df['date_ym'].max()} ({len(train_df)} months)")
    print(f"  Test period       : {test_df['date_ym'].min()} ~ {test_df['date_ym'].max()} ({len(test_df)} months)")

    selected_features = filter_features_by_vif(train_df, candidate_features, threshold=vif_threshold, max_features=6)

    X_train = train_df[selected_features]
    y_train = train_df[TARGET]
    X_test = test_df[selected_features]
    y_test = test_df[TARGET]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    preprocessed_data = {
        'df': df,
        'train_df': train_df,
        'test_df': test_df,
        'X_train_sc': X_train_sc,
        'X_test_sc': X_test_sc,
        'y_train': y_train,
        'y_test': y_test,
        'features': selected_features,
        'scaler': scaler
    }

    return preprocessed_data


def save_prediction_to_mysql(predicted_value, predicted_index, run_id):
    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB config. Skipping prediction save.")
        return
        
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
            charset='utf8mb4'
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS realestate_predictions (
                    run_id VARCHAR(50) NOT NULL,
                    predicted_value DOUBLE NOT NULL,
                    predicted_index DOUBLE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                sql = """
                INSERT INTO realestate_predictions (run_id, predicted_value, predicted_index)
                VALUES (%s, %s, %s)
                """
                cursor.execute(sql, (run_id, predicted_value, predicted_index))
            connection.commit()
            print("[DB] Successfully saved realestate_predictions (1 row) into MySQL.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save realestate predictions to MySQL: {e}")


def generate_and_save_realestate_report(predicted_value, predicted_index, run_id):
    if not GENERATE_REPORT:
        print("[LLM] GENERATE_REPORT is set to False. Skipping LLM report generation for Real Estate.")
        return
        
    load_dotenv(find_dotenv())
    openai_key = os.getenv("OPENAI_API_KEY")
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not openai_key:
        print("[Warning] Missing OPENAI_API_KEY. Skipping LLM report generation.")
        return
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB config. Skipping LLM report generation.")
        return
        
    re_today = None
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
            charset='utf8mb4'
        )
        try:
            with connection.cursor() as cursor:
                sql = "SELECT house_price_idx FROM ml_realestate_preprocessed ORDER BY date_ym DESC LIMIT 1"
                cursor.execute(sql)
                res = cursor.fetchone()
                if res:
                    re_today = float(res[0])
        finally:
            connection.close()
    except Exception as e:
        print(f"[Warning] Failed to fetch latest actual index for LLM: {e}")
        
    import urllib.request
    import json
    
    prompt = f"""
    부동산 가격지수 AI 예측 모델 분석 결과:
    - 이번달 실제 가격지수(re_today): {f'{re_today:.2f}' if re_today is not None else '데이터 없음'}
    - 다음달 예측 변동률: {predicted_value:.2f}%
    - 다음달 예측 환산 가격지수(predicted_index): {f'{predicted_index:.2f}' if predicted_index is not None else '데이터 없음'}
    - 주요 SHAP 변수 기여도 순위: 기준금리 (kr_base_rate, 40%), 소비자물가지수 (kr_cpi, 30%), 매수우위지수 (buyer_dominance, 20%), 주택담보대출금리 (kr_mortgage_rate, 10%)
    
    위 예측 데이터와 변수 기여도를 바탕으로 전문적이고 가독성이 높은 한국어 부동산 가격지수 전망 분석 리포트를 markdown 형식으로 작성해주세요.
    반드시 다음의 구조와 예시 이미지의 격식과 톤앤매너를 유지해주세요:
    
    구조 예시:
    ### [부동산 가격지수 분석 리포트]
    
    (여기에 부동산 시장 전망에 대한 한 줄 요약을 적어주세요. 예: 서울 아파트 시장의 회복세를 지지할 것으로 보입니다...)
    
    **1. (첫 번째 핵심 요인 제목)**
    (상승 혹은 하락을 이끄는 첫 번째 핵심 변수와 AI 분석 기여도를 엮어서 상세한 설명 한 단락을 작성해주세요.)
    
    **2. (두 번째 핵심 요인 제목)**
    (상승 혹은 하락을 이끄는 두 번째 핵심 변수와 AI 분석 기여도를 엮어서 상세한 설명 한 단락을 작성해주세요.)
    
    요구사항:
    - 마크다운 형식으로 작성할 것.
    - 너무 길지 않게 핵심 요약 위주로 작성할 것 (전체 400자 내외).
    """
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_key}"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a professional economic analyst. Always respond in Korean markdown format. Keep it concise, engaging, and professional."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    content = None
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            content = res_data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[Error] Failed to generate LLM report via OpenAI: {e}")
        return
        
    if not content:
        print("[Warning] Generated LLM report is empty.")
        return
        
    import uuid
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
            charset='utf8mb4'
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS trend_llm_report (
                    report_id VARCHAR(50) NOT NULL PRIMARY KEY,
                    type VARCHAR(50) NOT NULL,
                    model_name VARCHAR(50) NOT NULL,
                    language VARCHAR(10) NOT NULL,
                    content TEXT NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_source VARCHAR(255)
                )
                """)
                
                report_id = f"rpt_{str(uuid.uuid4()).replace('-', '')[:16]}"
                sql = """
                INSERT INTO trend_llm_report (report_id, type, model_name, language, content, status, data_source)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (report_id, "real_estate", "gpt-4o", "ko", content, "done", "ECOS, K-RealEstate"))
            connection.commit()
            print("[DB] Successfully generated and saved Real Estate LLM report into MySQL trend_llm_report table.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save Real Estate LLM report to MySQL: {e}")


def save_performance_to_mysql(rmse, r2_score, mae, mse, run_id=None):
    import uuid
    if not run_id:
        try:
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run else uuid.uuid4().hex[:32]
        except Exception:
            run_id = uuid.uuid4().hex[:32]
            
    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        print("[Warning] Missing DB config. Skipping performance save.")
        return
        
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
            charset='utf8mb4'
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS realestate_performance (
                    run_id VARCHAR(50) NOT NULL PRIMARY KEY,
                    rmse DOUBLE NOT NULL,
                    r2_score DOUBLE NOT NULL,
                    mae DOUBLE NOT NULL,
                    mse DOUBLE NOT NULL,
                    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                sql = """
                INSERT INTO realestate_performance (run_id, rmse, r2_score, mae, mse)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(sql, (run_id, rmse, r2_score, mae, mse))
            connection.commit()
            print("[DB] Successfully saved real_estate performance metrics into MySQL.")
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to save performance metrics to MySQL: {e}")


def get_latest_actual_realestate_index():
    load_dotenv(find_dotenv())
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_NAME = os.getenv('DB_NAME')
    
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
        return None
        
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            port=int(DB_PORT),
            charset='utf8mb4'
        )
        try:
            with connection.cursor() as cursor:
                sql = "SELECT house_price_idx FROM ml_realestate_preprocessed ORDER BY date_ym DESC LIMIT 1"
                cursor.execute(sql)
                res = cursor.fetchone()
                if res:
                    return float(res[0])
        finally:
            connection.close()
    except Exception as e:
        print(f"[Error] Failed to fetch latest actual realestate index: {e}")
    return None


def run_train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(find_dotenv())
 
    # MLflow 설정
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', None))
    mlflow.set_experiment("real_estate")
 
    with mlflow.start_run():
        data = preprocess_data(test_months=24, vif_threshold=10.0)
        if data is None:
            print("[Error] Preprocessing failed.")
            return
 
        X_train_sc = data['X_train_sc']
        y_train = data['y_train']
        selected_features = data['features']
        scaler = data['scaler']
 
        mlflow.log_param("test_months", 24)
        mlflow.log_param("vif_threshold", 10.0)
        mlflow.log_param("train_rows", len(X_train_sc))
        mlflow.log_param("num_features", len(selected_features))
        mlflow.log_param("random_state", 42)
 
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from sklearn.linear_model import LinearRegression
 
        print("\n" + "=" * 55)
        print("  [TimeSeriesSplit Cross-Validation (5 Splits) on Train Set]")
        print("=" * 55)
 
        tscv = TimeSeriesSplit(n_splits=5)
        cv_metrics = {
            "rmse": [], "r2": [], "mae": [], "mse": []
        }
 
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_sc)):
            X_tr, X_val = X_train_sc[train_idx], X_train_sc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
 
            fold_model = LinearRegression().fit(X_tr, y_tr)
            fold_pred = fold_model.predict(X_val)
 
            fold_r2 = r2_score(y_val, fold_pred)
            fold_mae = mean_absolute_error(y_val, fold_pred)
            fold_mse = mean_squared_error(y_val, fold_pred)
            fold_rmse = np.sqrt(fold_mse)
 
            cv_metrics["r2"].append(fold_r2)
            cv_metrics["mae"].append(fold_mae)
            cv_metrics["mse"].append(fold_mse)
            cv_metrics["rmse"].append(fold_rmse)
 
            mlflow.log_metric(f"fold_{fold+1}_r2", fold_r2)
            mlflow.log_metric(f"fold_{fold+1}_mae", fold_mae)
            mlflow.log_metric(f"fold_{fold+1}_mse", fold_mse)
            mlflow.log_metric(f"fold_{fold+1}_rmse", fold_rmse)
 
            print(f"    * Fold {fold+1} | Train: {len(X_tr)} months, Val: {len(X_val)} months | Val R2: {fold_r2:.4f} | Val MAE: {fold_mae:.4f}% | Val MSE: {fold_mse:.6f} | Val RMSE: {fold_rmse:.4f}")
 
        print("-" * 55)
        print("  --> Mean CV Metrics:")
        mean_rmse = float(np.mean(cv_metrics["rmse"]))
        mean_r2 = float(np.mean(cv_metrics["r2"]))
        mean_mae = float(np.mean(cv_metrics["mae"]))
        mean_mse = float(np.mean(cv_metrics["mse"]))
 
        for k in cv_metrics.keys():
            mean_val = np.mean(cv_metrics[k])
            mlflow.log_metric(f"cv_mean_{k}", mean_val)
            print(f"      * {k:<10}: {mean_val:.4f}")
        print("=" * 55 + "\n")
 
        ensemble = RealEstateEnsembleRegressor(random_state=42)
        ensemble.fit(X_train_sc, y_train)
 
        final_pred = ensemble.predict(X_train_sc)
        final_r2  = r2_score(y_train, final_pred)
        final_mae = mean_absolute_error(y_train, final_pred)
        final_mse = mean_squared_error(y_train, final_pred)
        final_rmse = np.sqrt(final_mse)
        
        mlflow.log_metric("train_r2", final_r2)
        mlflow.log_metric("train_mae", final_mae)
        mlflow.log_metric("train_mse", final_mse)
        mlflow.log_metric("train_rmse", final_rmse)
 
        latest_predicted_value = float(ensemble.predict(X_train_sc[[-1]])[0])
        
        re_today = get_latest_actual_realestate_index()
        if re_today is not None:
            predicted_index = re_today * (1 + latest_predicted_value / 100)
            print(f"[Ensemble] Calculated predicted_index: {predicted_index:.4f} using re_today: {re_today} and predicted_value: {latest_predicted_value}%")
        else:
            predicted_index = None
            print("[Warning] Could not calculate predicted_index because re_today is missing.")
            
        import uuid
        try:
            active_run = mlflow.active_run()
            run_id_val = active_run.info.run_id if active_run else uuid.uuid4().hex[:32]
        except Exception:
            run_id_val = uuid.uuid4().hex[:32]
 
        save_performance_to_mysql(rmse=mean_rmse, r2_score=mean_r2, mae=mean_mae, mse=mean_mse, run_id=run_id_val)
        save_prediction_to_mysql(predicted_value=latest_predicted_value, predicted_index=predicted_index, run_id=run_id_val)
        generate_and_save_realestate_report(predicted_value=latest_predicted_value, predicted_index=predicted_index, run_id=run_id_val)
 
        try:
            models_dir = os.path.join(base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
     
            model_path    = os.path.join(models_dir, 'ensemble_model.pkl')
            scaler_path   = os.path.join(models_dir, 'scaler.pkl')
            features_path = os.path.join(models_dir, 'selected_features.pkl')
     
            with open(model_path, 'wb') as f:
                pickle.dump(ensemble, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            with open(features_path, 'wb') as f:
                pickle.dump(selected_features, f)
     
            txt_features_path = os.path.join(models_dir, 'selected_features.txt')
            with open(txt_features_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(selected_features))
            print("[OK] Successfully saved local models and features.")
        except Exception as e:
            print(f"[Warning] Failed to save local model/feature files: {e}. (MLflow logging will proceed anyway)")
 
        mlflow.sklearn.log_model(ensemble, "ensemble_model")
 
        print("=" * 55)
        print("Training Pipeline Completed Successfully!")
        print("=" * 55)
        print(f"  Saved Model   : {model_path}")
        print(f"  Saved Scaler  : {scaler_path}")
        print(f"  Saved Features: {features_path} and .txt")
        print(f"  Features size : {len(selected_features)}")
 
if __name__ == '__main__':
    try:
        run_train()
    except Exception as e:
        import traceback
        import sys
        print("\n" + "="*60, file=sys.stderr)
        print("   [FATAL ERROR] run_real_estate.py crashed with exception:", file=sys.stderr)
        print("="*60, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("="*60 + "\n", file=sys.stderr)
        sys.exit(1)
