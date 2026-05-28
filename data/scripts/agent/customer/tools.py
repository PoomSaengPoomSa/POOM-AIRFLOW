import datetime
from .db import get_db_cursor

def get_portfolio_weight(customer_id: int):
    """
    Get customer asset portfolio details.
    """
    query = """
        SELECT c_id, name, total_assets, deposit, investment, pension, loan, net_worth, tendency, grade
        FROM customer
        WHERE c_id = %s
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        result = cursor.fetchone()
        return result

def search_today_news(date_str: str = None, keyword: str = None):
    """
    Search news archives. If no news is found for the specified date,
    returns the 10 most recent news articles as fallback.
    """
    if not date_str:
        date_str = datetime.date.today().strftime("%Y-%m-%d")

    params = []
    if keyword:
        query = """
            SELECT news_id, title, body, source, published_at
            FROM trend_news
            WHERE DATE(published_at) = %s AND (title LIKE %s OR body LIKE %s)
            ORDER BY published_at DESC
        """
        params = [date_str, f"%{keyword}%", f"%{keyword}%"]
    else:
        query = """
            SELECT news_id, title, body, source, published_at
            FROM trend_news
            WHERE DATE(published_at) = %s
            ORDER BY published_at DESC
        """
        params = [date_str]

    with get_db_cursor() as cursor:
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Fallback to 10 most recent news if today's news is empty
        if not results:
            if keyword:
                fallback_query = """
                    SELECT news_id, title, body, source, published_at
                    FROM trend_news
                    WHERE title LIKE %s OR body LIKE %s
                    ORDER BY published_at DESC
                    LIMIT 10
                """
                cursor.execute(fallback_query, (f"%{keyword}%", f"%{keyword}%"))
            else:
                fallback_query = """
                    SELECT news_id, title, body, source, published_at
                    FROM trend_news
                    ORDER BY published_at DESC
                    LIMIT 10
                """
                cursor.execute(fallback_query)
            results = cursor.fetchall()
            
        return results

def get_trend_report():
    """
    Retrieve completed trend reports:
    - For 'gold' (daily prediction): only fetch if created on the current date.
    - For other indicators (monthly predictions): fetch if created in the current month and year.
      If multiple exist, select the most recent one for each indicator type.
    """
    query = """
        (
            SELECT type, content, created_at
            FROM trend_llm_report
            WHERE type = 'gold' 
              AND status IN ('done', 'COMPLETED') 
              AND DATE(created_at) = CURDATE()
            ORDER BY created_at DESC
            LIMIT 1
        )
        UNION ALL
        (
            SELECT r1.type, r1.content, r1.created_at
            FROM trend_llm_report r1
            WHERE r1.status IN ('done', 'COMPLETED')
              AND r1.type != 'gold'
              AND YEAR(r1.created_at) = YEAR(CURDATE())
              AND MONTH(r1.created_at) = MONTH(CURDATE())
              AND r1.created_at = (
                  SELECT MAX(r2.created_at)
                  FROM trend_llm_report r2
                  WHERE r2.type = r1.type
                    AND r2.status IN ('done', 'COMPLETED')
                    AND YEAR(r2.created_at) = YEAR(CURDATE())
                    AND MONTH(r2.created_at) = MONTH(CURDATE())
              )
        )
    """
    with get_db_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
        return results

def get_customer_features(customer_id: int, months: int = 3):
    """
    Get customer features extracted from the database for the given period (months).
    """
    query = """
        SELECT category, contents, created_date
        FROM customer_information
        WHERE c_id = %s AND created_date >= DATE_SUB(NOW(), INTERVAL %s MONTH)
        ORDER BY created_date DESC
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id, months))
        results = cursor.fetchall()
        return results

def get_large_external_transactions(customer_id: int, threshold_amount: float = 10000000.0):
    """
    Retrieve external transactions where the customer transferred out a large amount of money.
    """
    query = """
        SELECT amount, opp_bank_name, briefs, ct_datetime, balance_after
        FROM customer_transaction
        WHERE c_id = %s AND opp_bank_name != '품' AND ct_type = 'W' AND amount >= %s
        ORDER BY ct_datetime DESC
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id, threshold_amount))
        results = cursor.fetchall()
        return results

def save_asset_insight(customer_id: int, insight: str):
    """
    Save the LLM generated asset profile analysis result to customer's llm_insight column.
    """
    query = """
        UPDATE customer
        SET llm_insight = %s
        WHERE c_id = %s
    """
    with get_db_cursor() as cursor:
        rows_affected = cursor.execute(query, (insight, customer_id))
        return rows_affected > 0

def save_churn_level(customer_id: int, grade: str, reason: str):
    """
    Insert a new churn risk level assessment into churn_level table.
    """
    query = """
        INSERT INTO churn_level (c_id, grade, reason, created_date)
        VALUES (%s, %s, %s, NOW())
    """
    with get_db_cursor() as cursor:
        rows_affected = cursor.execute(query, (customer_id, grade, reason))
        return rows_affected > 0

def get_recent_consultation_report(customer_id: int):
    """
    Get the latest consultation_report content for the customer.
    """
    query = """
        SELECT r.cr_id, r.content, m.consult_date, m.u_id
        FROM consultation_report r
        JOIN consultation_memo m ON r.cm_id = m.cm_id
        WHERE m.c_id = %s
        ORDER BY m.consult_date DESC
        LIMIT 1
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        result = cursor.fetchone()
        return result

def save_customer_feature(customer_id: int, category: str, contents: str):
    """
    Insert a new customer feature row into customer_information.
    """
    query = """
        INSERT INTO customer_information (c_id, category, contents, created_date)
        VALUES (%s, %s, %s, NOW())
    """
    with get_db_cursor() as cursor:
        rows_affected = cursor.execute(query, (customer_id, category, contents))
        return rows_affected > 0

def get_main_products():
    """
    Retrieve active bank main products from the product table.
    """
    query = """
        SELECT pd_id, name, explanation, type, features, target_customer, expected_return, return_type
        FROM product
        WHERE is_main = 1
    """
    with get_db_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
        return results

def save_product_matching(product_id: int, customer_id: int, is_suitable: int, reason: str):
    """
    Upsert product matching suitability evaluation result.
    To avoid piling up redundant rows for the same product-customer pair,
    we delete previous matching records for the pair first.
    """
    delete_query = """
        DELETE FROM product_matching
        WHERE pd_id = %s AND c_id = %s
    """
    insert_query = """
        INSERT INTO product_matching (pd_id, c_id, is_suitable, reason, created_date)
        VALUES (%s, %s, %s, %s, NOW())
    """
    with get_db_cursor() as cursor:
        cursor.execute(delete_query, (product_id, customer_id))
        rows_affected = cursor.execute(insert_query, (product_id, customer_id, is_suitable, reason))
        return rows_affected > 0

def get_customer_relationship(customer_id: int):
    """
    Retrieve customer family relationships from customer_relationship table.
    """
    query = """
        SELECT relationship, birthday, job, is_spouse, wedding_date
        FROM customer_relationship
        WHERE c_id = %s
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        results = cursor.fetchall()
        return results

def get_customer_active_products(customer_id: int):
    """
    Retrieve products currently held by the customer.
    """
    query = """
        SELECT cp.pd_id, p.name as product_name, cp.opening_date, cp.expiration_date
        FROM customer_product cp
        JOIN product p ON cp.pd_id = p.pd_id
        WHERE cp.c_id = %s
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        results = cursor.fetchall()
        return results

def get_customer_accounts(customer_id: int):
    """
    Retrieve customer's account types and balances.
    """
    query = """
        SELECT account_num, account_type, balance, opening_date
        FROM customer_account
        WHERE c_id = %s
    """
    with get_db_cursor() as cursor:
        cursor.execute(query, (customer_id,))
        results = cursor.fetchall()
        return results
