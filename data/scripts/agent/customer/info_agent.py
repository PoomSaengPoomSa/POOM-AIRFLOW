import os
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain and LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from . import tools

# Absolute path resolution to strictly load .env from agent/customer/.env
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Map LANGSMITH_ environment variables to LANGCHAIN_ standard tracing variables
if os.getenv("LANGSMITH_TRACING") == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
if os.getenv("LANGSMITH_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")
if os.getenv("LANGSMITH_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT").strip('"\'')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        f"Configuration error: OPENAI_API_KEY is missing from the .env file. "
        f"Please verify your local .env file at {env_path}"
    )

# Global default model configuration
DEFAULT_MODEL = "gpt-4o-mini"

# 1. Helper function to dynamically load prompts from markdown files
def load_prompt(filename: str) -> str:
    """
    Utility to load prompt templates from the prompt directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "prompt", filename)
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

# 2. Pydantic Model for Churn Structured Output
class ChurnAssessment(BaseModel):
    grade: str = Field(
        description="이탈 위험 등급. 반드시 '양호', '주의', '위험' 중 하나여야 합니다."
    )
    reason: str = Field(
        description="판정 사유. 반드시 공백 포함 80자 이내의 한 문장(한국어 경어체)으로 간결하게 작성해 주세요. (VARCHAR(100) 길이 제한이 있으므로 반드시 90자 이하로만 작성해야 합니다.)"
    )

# 3. Pydantic Model for Dynamic Tool Selection
class ToolSelection(BaseModel):
    call_search_today_news: bool = Field(description="오늘자 금융/경제 뉴스를 검색할지 여부. 고객이 적극투자형이거나 자산 비중에서 투자가 높은 경우 시장 정보 수집을 위해 참(True)으로 설정합니다.")
    news_keyword: Optional[str] = Field(description="뉴스 검색 키워드 (예: 금리, 금, 부동산, 주식, 코스피 등), 필요 없으면 None")
    call_get_trend_report: bool = Field(description="경제 지표 트렌드 보고서(금값, 기준금리, 부동산)를 가져올지 여부. 고객의 연금, 투자, 대출 자산이 존재하고 거시 경제 지표와의 매칭 진단이 유용할 때 참(True)으로 설정합니다.")
    call_get_customer_features: bool = Field(description="고객의 최근 3개월 특징 기록을 조회할지 여부. 고객 유지 및 행동 성향 파악을 위해 참(True)으로 설정합니다.")
    call_get_large_external_transactions: bool = Field(description="고객의 타행 거액 송금 내역을 조회할지 여부. 자산이 고액이거나 대출이 있어 이탈 징후 분석이 필요할 때 참(True)으로 설정합니다.")
    transaction_threshold: Optional[float] = Field(description="거액 송금 조회 기준 금액 (원 단위, 기본값 10,000,000원), 필요 없으면 None", default=10000000.0)
    reason: str = Field(description="해당 데이터 수집 도구들을 선택한 이유에 대한 에이전트의 구체적인 분석 및 판단 근거 (한 문장)")

# 4. State Definition for LangGraph
class AgentState(TypedDict):
    customer_id: int
    portfolio: Optional[Dict[str, Any]]
    tool_selection: Optional[Dict[str, Any]]
    today_news: Optional[List[Dict[str, Any]]]
    trend_reports: Optional[List[Dict[str, Any]]]
    recent_features: Optional[List[Dict[str, Any]]]
    large_transactions: Optional[List[Dict[str, Any]]]
    asset_insight: Optional[str]
    churn_grade: Optional[Optional[str]]
    churn_reason: Optional[Optional[str]]
    errors: List[str]

# 5. Graph Node Implementations

def load_basic_profile_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Load basic customer profile and asset weights.
    """
    customer_id = state["customer_id"]
    errors = list(state.get("errors", []))
    
    try:
        portfolio = tools.get_portfolio_weight(customer_id)
        if not portfolio:
            raise ValueError(f"Customer with ID {customer_id} not found in database.")
        
        return {
            "portfolio": portfolio,
            "errors": errors
        }
    except Exception as e:
        errors.append(f"load_basic_profile failed: {str(e)}")
        return {"errors": errors}

def determine_tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: AI Agent judges and determines which tools are necessary for this specific customer.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    portfolio = state["portfolio"]
    customer_id = state["customer_id"]

    # System/User Prompts for Tool Routing
    system_prompt = (
        "당신은 고객의 기본 프로필과 자산 배분 비중을 정교하게 분석하여, "
        "해당 고객에게 맞춤형 자산 리포트 및 이탈 위험 진단을 내리기 위해 추가로 수집해야 할 금융 정보/경제 지표 도구를 판단하는 AI 자산배분 분석가이자 데이터 라우터입니다.\n\n"
        "현재 사용 가능한 추가 데이터 조회 도구 목록:\n"
        "1. `search_today_news` (오늘의 뉴스): 투자 비중이 높거나 적극형 성향인 고객의 시장 트렌드 매칭 분석용. 적합한 키워드 지정 가능.\n"
        "2. `get_trend_report` (경제 지표 분석 보고서): 기준금리, 금값, 부동산 등 주요 거시 지표 변화가 고객 자산(투자, 연금, 대출 등)에 미칠 영향 분석용.\n"
        "3. `get_customer_features` (최근 3개월 고객 특징): 평소 금융 성향이나 상담 중 언급된 이탈 위험 징후를 추적하기 위해 필수.\n"
        "4. `get_large_external_transactions` (타행 거액 송금): 총자산이 억대 이상으로 고액이거나 대출이 있는 경우 타 금융사로의 유출 내역을 감지하기 위한 이탈 위험 평가용. 기준 금액 설정 가능.\n\n"
        "고객의 상황에 꼭 필요한 도구들만 선별적으로 선택(True/False)하고, 그 근거를 도출하십시오."
    )

    user_prompt = (
        f"## 분석 대상 고객 프로필\n"
        f"- 고객명: {portfolio['name']}\n"
        f"- 등급: {portfolio['grade']}\n"
        f"- 투자성향: {portfolio['tendency']}\n"
        f"- 총자산: {portfolio['total_assets']:,}원\n"
        f"  - 예금: {portfolio['deposit']:,}원 (순자산 대비 비중: {portfolio['deposit']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 투자: {portfolio['investment']:,}원 (순자산 대비 비중: {portfolio['investment']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 연금: {portfolio['pension']:,}원 (순자산 대비 비중: {portfolio['pension']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 대출: {portfolio['loan']:,}원 (순자산 대비 부채비율: {portfolio['loan']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 순자산: {portfolio['net_worth']:,}원\n\n"
        f"이 고객을 심층 분석하기 위해 추가 수집해야 할 정보 도구들을 판단해 주십시오."
    )

    try:
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ToolSelection)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt)
        ])

        chain = prompt | structured_llm
        selection: ToolSelection = chain.invoke({})
        
        return {
            "tool_selection": selection.dict()
        }
    except Exception as e:
        errors.append(f"determine_tools failed: {str(e)}")
        return {"errors": errors}

def execute_selected_tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Execute the chosen tools dynamically.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    customer_id = state["customer_id"]
    tool_selection = state["tool_selection"]

    today_news = []
    trend_reports = []
    recent_features = []
    large_transactions = []

    try:
        # Call news tool dynamically
        if tool_selection.get("call_search_today_news"):
            keyword = tool_selection.get("news_keyword")
            print(f"   [Tool Run] Running search_today_news(keyword='{keyword}')")
            today_news = tools.search_today_news(keyword=keyword)
        else:
            print("   [Tool Skip] Skipping search_today_news based on AI judgment.")

        # Call trend reports tool dynamically
        if tool_selection.get("call_get_trend_report"):
            print("   [Tool Run] Running get_trend_report()")
            trend_reports = tools.get_trend_report()
        else:
            print("   [Tool Skip] Skipping get_trend_report based on AI judgment.")

        # Call customer features tool dynamically
        if tool_selection.get("call_get_customer_features"):
            print("   [Tool Run] Running get_customer_features(months=3)")
            recent_features = tools.get_customer_features(customer_id, months=3)
        else:
            print("   [Tool Skip] Skipping get_customer_features based on AI judgment.")

        # Call large transactions tool dynamically
        if tool_selection.get("call_get_large_external_transactions"):
            threshold = tool_selection.get("transaction_threshold") or 10000000.0
            print(f"   [Tool Run] Running get_large_external_transactions(threshold={threshold:,}원)")
            large_transactions = tools.get_large_external_transactions(customer_id, threshold_amount=threshold)
        else:
            print("   [Tool Skip] Skipping get_large_external_transactions based on AI judgment.")

        return {
            "today_news": today_news,
            "trend_reports": trend_reports,
            "recent_features": recent_features,
            "large_transactions": large_transactions
        }
    except Exception as e:
        errors.append(f"execute_selected_tools failed: {str(e)}")
        return {"errors": errors}

def analyze_assets_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: LangChain LLM call to analyze portfolio assets and trends.
    Uses prompts dynamically from prompt/asset_analysis_system.md and prompt/asset_analysis_user.md.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    portfolio = state["portfolio"]
    today_news = state["today_news"]
    trend_reports = state["trend_reports"]

    # Format input data
    portfolio_str = (
        f"- 고객명: {portfolio['name']}\n"
        f"- 투자성향: {portfolio['tendency']}\n"
        f"- 등급: {portfolio['grade']}\n"
        f"- 총자산: {portfolio['total_assets']:,}원\n"
        f"  - 예금: {portfolio['deposit']:,}원 (순자산 대비 비중: {portfolio['deposit']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 투자: {portfolio['investment']:,}원 (순자산 대비 비중: {portfolio['investment']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 연금: {portfolio['pension']:,}원 (순자산 대비 비중: {portfolio['pension']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 대출: {portfolio['loan']:,}원 (순자산 대비 부채비율: {portfolio['loan']/max(1, portfolio['net_worth'])*100:.1f}%)\n"
        f"  - 순자산: {portfolio['net_worth']:,}원\n"
    )

    # Format dynamically fetched information (explain if skipped)
    if state["tool_selection"].get("call_search_today_news"):
        news_list = []
        for n in today_news[:5]:
            news_list.append(f"[{n['source']}] {n['title']}\n{n['body'][:200]}...")
        news_str = "\n\n".join(news_list) if news_list else "당일 수집된 주요 뉴스 없음."
    else:
        news_str = "[참고] 에이전트의 수집 판단 제외: 해당 고객의 포트폴리오 성격상 당일 뉴스의 직접적인 필요성이 낮아 분석 데이터에서 제외되었습니다."

    if state["tool_selection"].get("call_get_trend_report"):
        reports_list = []
        for r in trend_reports:
            reports_list.append(f"[{r['type'].upper()} 트렌드 분석 보고서]\n{r['content']}")
        reports_str = "\n\n".join(reports_list) if reports_list else "활성화된 지표 트렌드 분석 보고서 없음."
    else:
        reports_str = "[참고] 에이전트의 수집 판단 제외: 금값/금리/부동산 거시 지표 변화 영향도가 낮아 분석 데이터에서 제외되었습니다."

    try:
        # Load enhanced prompt markdown files dynamically
        system_prompt = load_prompt("asset_analysis_system.md")
        user_prompt_template = load_prompt("asset_analysis_user.md")

        # Initialize LangChain ChatModel
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.7, api_key=OPENAI_API_KEY)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt_template)
        ])

        chain = prompt | llm
        response = chain.invoke({
            "portfolio_str": portfolio_str,
            "news_str": news_str,
            "reports_str": reports_str,
            "tendency": portfolio['tendency']
        })
        
        return {"asset_insight": response.content.strip()}
    except Exception as e:
        errors.append(f"analyze_assets failed: {str(e)}")
        return {"errors": errors}

def analyze_churn_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 5: LangChain ChatModel call with Structured Output for Churn Assessment.
    Loads prompts dynamically from prompt/churn_risk_system.md and prompt/churn_risk_user.md.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    portfolio = state["portfolio"]
    recent_features = state["recent_features"]
    large_transactions = state["large_transactions"]

    # Format dynamically fetched information (explain if skipped)
    if state["tool_selection"].get("call_get_customer_features"):
        features_list = []
        for f in recent_features:
            features_list.append(f"[{f['category']} - {f['created_date'].strftime('%Y-%m-%d')}] {f['contents']}")
        features_str = "\n".join(features_list) if features_list else "최근 3개월 내 기록된 고객 특징 없음."
    else:
        features_str = "[참고] 에이전트의 수집 판단 제외: 고객 행동 특징 분석이 불필요하다고 AI가 진단하여 이력 제외함."

    if state["tool_selection"].get("call_get_large_external_transactions"):
        tx_list = []
        for t in large_transactions[:5]:
            tx_list.append(
                f"- 일시: {t['ct_datetime'].strftime('%Y-%m-%d %H:%M:%S')}, "
                f"금액: {t['amount']:,}원, "
                f"상대행: {t['opp_bank_name']}, "
                f"적요: {t['briefs']}, "
                f"거래후잔액: {t['balance_after']:,}원"
            )
        tx_str = "\n".join(tx_list) if tx_list else "최근 타행 거액 송금 이력 없음."
    else:
        tx_str = "[참고] 에이전트의 수집 판단 제외: 거액 송금 유출 패턴 분석 불필요로 판단하여 거래 이력 제외함."

    try:
        # Load enhanced prompt markdown files dynamically
        system_prompt = load_prompt("churn_risk_system.md")
        user_prompt_template = load_prompt("churn_risk_user.md")

        # Initialize LangChain model with Structured Output matching Pydantic schema
        llm = ChatOpenAI(model=DEFAULT_MODEL, temperature=0.3, api_key=OPENAI_API_KEY)
        structured_llm = llm.with_structured_output(ChurnAssessment)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt_template)
        ])

        chain = prompt | structured_llm
        assessment: ChurnAssessment = chain.invoke({
            "name": portfolio["name"],
            "grade": portfolio["grade"],
            "total_assets": portfolio["total_assets"],
            "deposit": portfolio["deposit"],
            "loan": portfolio["loan"],
            "features_str": features_str,
            "tx_str": tx_str
        })

        # Double check constraints programmatically
        grade = assessment.grade.strip()
        grade_map = {
            "Low": "양호", "Medium": "주의", "High": "위험",
            "low": "양호", "medium": "주의", "high": "위험"
        }
        grade = grade_map.get(grade, grade)
        if grade not in ["양호", "주의", "위험"]:
            grade = "양호"

        reason = assessment.reason.strip()
        if len(reason) > 100:
            reason = reason[:97] + "..."

        return {
            "churn_grade": grade,
            "churn_reason": reason
        }
    except Exception as e:
        errors.append(f"analyze_churn failed: {str(e)}")
        return {"errors": errors}

def save_results_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 6: Save LLM results back to DB via tools.
    """
    errors = list(state.get("errors", []))
    if errors:
        return {}

    customer_id = state["customer_id"]
    asset_insight = state["asset_insight"]
    churn_grade = state["churn_grade"]
    churn_reason = state["churn_reason"]

    try:
        # Save asset insights
        insight_saved = tools.save_asset_insight(customer_id, asset_insight)
        if not insight_saved:
            raise ValueError(f"Failed to update asset insight in customer table for ID {customer_id}")

        # Save churn risk assessment
        churn_saved = tools.save_churn_level(customer_id, churn_grade, churn_reason)
        if not churn_saved:
            raise ValueError(f"Failed to insert churn risk level into churn_level table for ID {customer_id}")

        print(f"  [+] Data saved successfully in database for customer {customer_id}")
        return {}
    except Exception as e:
        errors.append(f"save_results failed: {str(e)}")
        return {"errors": errors}


# 6. Compiled State Graph Construction
workflow = StateGraph(AgentState)

# Add nodes to graph
workflow.add_node("load_basic_profile", load_basic_profile_node)
workflow.add_node("determine_tools", determine_tools_node)
workflow.add_node("execute_selected_tools", execute_selected_tools_node)
workflow.add_node("analyze_assets", analyze_assets_node)
workflow.add_node("analyze_churn", analyze_churn_node)
workflow.add_node("save_results", save_results_node)

# Set entry point
workflow.set_entry_point("load_basic_profile")

# Add transition edges
workflow.add_edge("load_basic_profile", "determine_tools")
workflow.add_edge("determine_tools", "execute_selected_tools")
workflow.add_edge("execute_selected_tools", "analyze_assets")
workflow.add_edge("analyze_assets", "analyze_churn")
workflow.add_edge("analyze_churn", "save_results")
workflow.add_edge("save_results", END)

# Compile graph
compiled_app = workflow.compile()


class CustomerInfoAgent:
    """
    Customer Info Agent (고객 정보 분석 에이전트)
    Analyzes asset allocations and calculates customer churn risk.
    """
    def __init__(self, model_name: str = None):
        global DEFAULT_MODEL
        if model_name:
            DEFAULT_MODEL = model_name
        self.app = compiled_app

    def run(self, customer_id: int) -> Dict[str, Any]:
        """
        Run the complete compiled LangGraph workflow for the given customer ID.
        """
        initial_state: AgentState = {
            "customer_id": customer_id,
            "portfolio": None,
            "tool_selection": None,
            "today_news": None,
            "trend_reports": None,
            "recent_features": None,
            "large_transactions": None,
            "asset_insight": None,
            "churn_grade": None,
            "churn_reason": None,
            "errors": []
        }
        
        final_state = self.app.invoke(
            initial_state,
            config={"run_name": "CustomerInfoAgent", "tags": ["info_agent"]}
        )
        
        if final_state.get("errors"):
            raise RuntimeError(f"LangGraph execution encountered errors in CustomerInfoAgent: {final_state['errors']}")
            
        return final_state

    def analyze_assets(self, customer_id: int) -> str:
        res = self.run(customer_id)
        return res["asset_insight"]

    def analyze_churn_risk(self, customer_id: int) -> Dict[str, Any]:
        res = self.run(customer_id)
        return {
            "grade": res["churn_grade"],
            "reason": res["churn_reason"]
        }
