# ==========================================
# Poom Customer Info Agent Runner
# ==========================================
DEFAULT_MODEL = "gpt-4o-mini"

import argparse
import os
import sys
from dotenv import load_dotenv

# Ensure import works correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # POOM-AI root
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from agent.customer.db import get_db_cursor
from agent.customer.info_agent import CustomerInfoAgent

def get_all_customer_ids():
    """
    Fetch all customer IDs from the customer table.
    """
    query = "SELECT c_id FROM customer"
    with get_db_cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
        return [row["c_id"] for row in results]

def main():
    parser = argparse.ArgumentParser(description="Poom Customer Info Agent (Agent 1) Runner")
    parser.add_argument(
        "--c_id", 
        type=str, 
        help="Comma-separated list of customer IDs (e.g., 1001,1002,1003). If omitted, runs for all customers."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL, 
        help=f"OpenAI LLM model to use (default: {DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    
    # Resolve customer IDs
    if args.c_id:
        try:
            customer_ids = [int(cid.strip()) for cid in args.c_id.split(",") if cid.strip()]
            print(f"[*] Running analysis for specified customer IDs: {customer_ids}")
        except ValueError:
            print("[-] Error: --c_id must be a comma-separated list of integers.")
            sys.exit(1)
    else:
        print("[*] No customer IDs provided. Fetching all customers from database...")
        customer_ids = get_all_customer_ids()
        print(f"[*] Found {len(customer_ids)} customers in database.")

    if not customer_ids:
        print("[-] No customers to analyze.")
        sys.exit(0)

    try:
        info_agent = CustomerInfoAgent(model_name=args.model)
    except Exception as e:
        print(f"[-] Failed to initialize CustomerInfoAgent: {e}")
        sys.exit(1)

    print(f"\n==================================================")
    print(f"  Starting Customer Info Agent (Agent 1)")
    print(f"  Model: {args.model}")
    print(f"  Total Customers: {len(customer_ids)}")
    print(f"==================================================\n")

    success_count = 0
    fail_count = 0

    for idx, c_id in enumerate(customer_ids, 1):
        print(f"[{idx}/{len(customer_ids)}] Processing Customer ID: {c_id} ...")
        try:
            print(f"  -> Executing LangGraph workflow...")
            result = info_agent.run(c_id)
            print(f"     [Success] Asset insight saved. (Length: {len(result['asset_insight'])} characters)")
            print(f"     [Success] Churn risk assessed: {result['churn_grade']}")
            print(f"     Reason: {result['churn_reason']}")
            success_count += 1
            print(f"  [+] Customer {c_id} completed successfully.\n")
        except Exception as e:
            fail_count += 1
            print(f"  [-] [Error] Failed to process Customer {c_id}: {e}\n")

    print(f"==================================================")
    print(f"  Execution Complete!")
    print(f"  Successfully processed: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"==================================================")

if __name__ == "__main__":
    main()
