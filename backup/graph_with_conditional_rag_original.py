import os
import json
import asyncio
import httpx
import logging
import faiss
import numpy as np
from typing import Annotated, List, Dict, Any, TypedDict, Union
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFaceEmbeddings

# temp.py에서 요약 메뉴얼과 카테고리별 상세 정보를 가져옵니다.
from temp import SHORT_MANUAL, CATEGORY_DICT, SHORTER_MANUAL

# 환경 변수 로드
load_dotenv()
# 기본 로깅은 끄고 커스텀 출력만 봅니다
logging.basicConfig(level=logging.CRITICAL)

# -------------------------------------------------------------------------
# 0. VectorDB 관련 설정
# -------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "faiss_db.json"
DB_PERMANENT_PATH = DATA_DIR / "faiss_db_permanent.json"
INDEX_PATH = DATA_DIR / "faiss_index.bin"
embedding_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

def load_db(file_path):
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_db(file_path, db):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=4)

def load_faiss_index():
    if INDEX_PATH.exists() and INDEX_PATH.stat().st_size > 0:
        try:
            with open(INDEX_PATH, "rb") as f:
                data = f.read()
            return faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
        except Exception as e:
            print(f"Error loading FAISS index, creating new one: {e}")
            return faiss.IndexFlatL2(768)
    return faiss.IndexFlatL2(768)

def save_faiss_index(index):
    index_data = faiss.serialize_index(index)
    with open(INDEX_PATH, "wb") as f:
        f.write(index_data)

db = load_db(DB_PATH)
db_permanent = load_db(DB_PERMANENT_PATH)
index = load_faiss_index()

# -------------------------------------------------------------------------
# 1. 디버깅 및 로깅 유틸리티
# -------------------------------------------------------------------------

class DebugCallbackHandler(BaseCallbackHandler):
    def __init__(self, show_prompt: bool = True, show_token: bool = True):
        self.show_prompt = show_prompt
        self.show_token = show_token

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> None:
        if not self.show_prompt: return
        print("\n" + "🔵 " * 10 + " [LLM INPUT PROMPT] " + "🔵 " * 10)
        for msg in messages[0]:
            content = str(msg.content)
            if len(content) > 1500:
                display_content = content[:500] + "\n... (중략: 매뉴얼 본문) ...\n" + content[-500:]
            else:
                display_content = content
            print(f"[{msg.type.upper()}]: {display_content}")
        print("🔵 " * 25 + "\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:

        if not self.show_token:
            return
        
        try:
            generation = response.generations[0][0]
            usage = {}
            if hasattr(generation, 'message') and hasattr(generation.message, 'response_metadata'):
                usage = generation.message.response_metadata.get('token_usage', {}) or generation.message.response_metadata.get('usage', {})
            elif hasattr(response, 'llm_output') and response.llm_output:
                usage = response.llm_output.get('token_usage', {})

            if usage:
                input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
                output_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
                total = usage.get('total_tokens', 0) or (input_tokens + output_tokens)
                print(f"📊 [TOKEN USAGE] Input: {input_tokens} | Output: {output_tokens} | Total: {total}")
            else:
                print("📊 [TOKEN USAGE] 메타데이터 없음")
        except Exception as e:
            print(f"📊 [TOKEN USAGE] 파싱 실패: {e}")


def print_state_debug(node_name: str, state: Dict):
    print(f"\n" + "⚡ " * 15)
    print(f"⚡ [STATE DUMP] Node: {node_name}")
    print(f"⚡ " * 15)
    keys_to_show = ["current_stage_index", "api_results", "final_answer", "documents"]
    if "endpoint_plan" in state and state["endpoint_plan"]:
        print(f"  - endpoint_plan: {len(state['endpoint_plan'])} stages loaded")
    if "executable_stage" in state and state["executable_stage"]:
        print(f"  - executable_stage: {len(state['executable_stage'].get('steps', []))} steps ready")
    for k in keys_to_show:
        if k in state and state[k]:
            val = state[k]
            if k == "documents":
                 print(f"  - documents: (Content loaded, length: {len(val)})")
            elif k == "api_results":
                print(f"  - api_results: {list(val.keys())}")
            else:
                print(f"  - {k}: {val}")
    print("-" * 60 + "\n")

# -------------------------------------------------------------------------
# 2. 설정 및 데이터 모델
# -------------------------------------------------------------------------

JSON_PATH = Path(__file__).resolve().parent.parent / "backend" / "manual" / "uri_params.json"
PARAMS_JSON = load_db(JSON_PATH)

# --- 출력 모델 ---
class ApiStep(BaseModel):
    step_id: str = Field(description="단계 식별자 (예: '1-1', '1-2')")
    endpoint: str = Field(description="호출할 API 엔드포인트")
    reasoning: str = Field(description="이 단계의 실행 목적")

class ParallelStage(BaseModel):
    stage_id: int = Field(description="실행 순서 (1부터 시작)")
    steps: List[ApiStep] = Field(description="이 단계에서 병렬로 실행할 엔드포인트 리스트")

class EndpointExecutionPlan(BaseModel):
    plan: List[ParallelStage] = Field(description="순차적으로 실행될 스테이지 리스트")

class ExecutableStep(BaseModel):
    step_id: str = Field(description="고유 식별자 (예: '3-1-instance-1')")
    endpoint: str = Field(description="호출할 API 엔드포인트")
    params: Dict[str, Any] = Field(description="API 호출에 사용할 실제 파라미터")
    reasoning: str = Field(description="이 특정 API 호출을 실행하는 이유")

class ExecutableStage(BaseModel):
    steps: List[ExecutableStep] = Field(description="이번 스테이지에서 병렬로 실행할 API 호출 목록")

# --- 카테고리 선택 모델 (새로 추가) ---
class CategorySelection(BaseModel):
    categories: List[str] = Field(description="사용자 질문과 관련된 카테고리 이름 목록")


# --- 메인 Agent State (수정) ---
class AgentState(TypedDict):
    question: str
    documents: str  # RAG 실패 시 API 명세서를 담을 필드
    endpoint_plan: List[Dict] 
    executable_stage: Dict 
    current_stage_index: int
    api_results: Dict
    final_answer: str
    from_db: bool # DB에서 왔는지 여부

# -------------------------------------------------------------------------
# 3. 도구(Tool) 및 에이전트 클래스
# -------------------------------------------------------------------------

async def call_torus_api(endpoint: str, params: Dict[str, Any] = {}) -> Dict:
    base_url = "http://127.0.0.1:8000"
    url = f"{base_url}{endpoint}"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=10.0)
            if response.status_code == 200:
                print(f"   📥 [API Response] Success from {url} | Params: {params}")
                return response.json()
            else:
                print(f"   ⚠️ [API Response] Failed from {url} | Status: {response.status_code}")
                return {"error": f"Status {response.status_code}", "body": response.text}
    except Exception as e:
        print(f"   🔥 [API Response] Exception from {url} | Error: {e}")
        return {"error": str(e)}

class TorusDynamicAgent:
    def __init__(self, model):
        self.debug_handler = DebugCallbackHandler()
        self.model = model
        self.category_selector_model = model.with_structured_output(CategorySelection)
        self.endpoint_planner_model = model.with_structured_output(EndpointExecutionPlan)
        self.param_planner_model = model.with_structured_output(ExecutableStage)
        self.PARAMS_JSON = PARAMS_JSON

    async def get_params_info(self, endpoint_list: List[str]):
        results = {}
        for endpoint in endpoint_list:
            endpoint_info = self.PARAMS_JSON.get(endpoint)
            results[endpoint] = endpoint_info.get("required_params") if endpoint_info else "존재하지 않는 엔드포인트"
        return results
    
    # --- 노드 정의 ---
    
    async def vector_store_node(self, state: AgentState):
        print_state_debug("Vector Store", state)
        question = state["question"]
        
        if index.ntotal > 0:
            query_embedding = np.array([embedding_model.embed_query(question)], dtype='float32')
            distances, indices = index.search(query_embedding, 1)
            
            print(f"유사도 계산 결과 : {distances[0][0]}")
            if distances[0][0] < 10:
                matched_question = list(db.keys())[indices[0][0]]
                print(f"🔍 Found similar question in DB: '{matched_question}'")
                return {
                    "endpoint_plan": db[matched_question],
                    "from_db": True, "documents": "",
                    "current_stage_index": 0, "api_results": {}
                }

        print("🚫 No similar question found in DB. Proceeding to Category Selection.")
        return {"from_db": False, "documents": ""}

    async def category_selection_node(self, state: AgentState):
        print_state_debug("Category Selection (Temp Node)", state)
        question = state["question"]
        
        prompt = (
            "당신은 사용자 질문의 핵심 의도를 파악하여, 질문 해결에 필요한 API 카테고리를 정확히 식별하는 전문가입니다.\n"
            "주어진 질문의 목표를 달성하기 위해 아래 JSON 형식의 카테고리 목록에서 **관련된 모든 카테고리**를 신중하게 선택하세요.\n\n"
            f"### 사용자 질문:\n{question}\n\n"
            f"### 전체 카테고리 목록 (JSON 형식):\n{SHORT_MANUAL}\n\n"
            "분석 후, 관련된 카테고리들의 `category` 필드 값만 JSON 리스트 형태로 반환해주세요."
        )
        
        selection_obj = await self.category_selector_model.ainvoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": [self.debug_handler]}
        )
        
        selected_categories = selection_obj.categories
        print(f"\n🧠 LLM selected categories: {selected_categories}")
        
        # 선택된 카테고리의 API 명세서를 수집
        relevant_docs = []
        for cat in selected_categories:
            if cat in CATEGORY_DICT:
                relevant_docs.append(f"--- Category: {cat} ---\n{CATEGORY_DICT[cat]}")
        
        documents_str = "\n\n".join(relevant_docs)
        print("📚 Assembled relevant API documents for the planner.")
        
        return {"documents": documents_str}

    async def endpoint_planner_node(self, state: AgentState):
        print_state_debug("Endpoint Planner", state)
        question = state["question"]
        documents = state["documents"] # RAG 실패 시 category_selection_node가 채워줌

        if not documents: # 비어있으면 에러 방지용 (이론상 여기에 도달하면 안됨)
            return {"endpoint_plan": [], "current_stage_index": 0, "api_results": {}}
            
        system_text = (
            "당신은 TORUS API 병렬 처리 설계자입니다. **주어진 API 명세서만을 참고하여** 실행 계획을 수립해야 합니다."
            "서로 의존성이 없는 API 호출은 같은 스테이지(Stage)로 묶어야 합니다.\n\n"
            "=== 작성 규칙 ===\n"
            "0. **machine list**: 모든 질문은 /machine/list (필요 파라미터 없음) 요청이 필수적입니다. 첫 번째 stage에 포함시키세요.\n"
            "1. **Parallel Stage**: 이전 단계의 결과가 필요하지 않고, 서로 독립적인 API들은 하나의 Stage 안에 리스트로 넣으세요.\n"
            "2. **Sequential Stage**: 이전 단계의 데이터가 반드시 필요한 경우에만 다음 스테이지로 분리하세요.\n\n"


            f"[사용 가능한 API 명세서]\n{documents}"
        )
        plan_obj = await self.endpoint_planner_model.ainvoke(
            [SystemMessage(content=system_text), HumanMessage(content=question)],
            config={"callbacks": [self.debug_handler]}
        )
        plan_stages = [stage.model_dump() for stage in plan_obj.plan]
        all_endpoints = [step['endpoint'] for stage in plan_stages for step in stage['steps']]
        params_map = await self.get_params_info(all_endpoints)

        print("\n📋 [High-Level Endpoint Plan]")
        for stage in plan_stages:
            print(f"   Stage {stage['stage_id']}")
            for step in stage['steps']:
                ep = step['endpoint']
                step['lookup_params'] = params_map.get(ep, "정보 없음")
                print(f"     - [{step['step_id']}] {ep} | Required Params: {step['lookup_params']}")
        return {"endpoint_plan": plan_stages, "current_stage_index": 0, "api_results": {}}

    async def param_planner_node(self, state: AgentState):
        print_state_debug("Param Planner", state)
        current_idx = state["current_stage_index"]
        endpoint_plan = state["endpoint_plan"]
        if not endpoint_plan or current_idx >= len(endpoint_plan):
            return {"executable_stage": {"steps": []}}
        
        current_stage_plan = endpoint_plan[current_idx]
        stage_id = current_stage_plan['stage_id']
        api_results = state.get("api_results", {})
        question = state["question"]
        print(f"\n🧠 [Planning Params for Stage {stage_id}]")
        
        prompt = (
            "당신은 API 파라미터를 동적으로 결정하는 '파라미터 플래너'입니다.\n"
            "주어진 '다음 실행 계획'과 '이전 데이터'를 바탕으로, 이번에 실행할 API 호출 목록을 '완성'하세요.\n\n"
            "## 중요 규칙\n"
            "0. **파라미터 값 규칙**: 모든 파라미터 값은 1부터 시작합니다. 0은 사용하지 않습니다.\n"
            "1. **반복 실행**: 만약 이전 데이터에 리스트(예: 장비 20개)가 있고, 다음 계획이 그 리스트의 각 항목에 대해 실행되어야 한다면, 리스트의 모든 항목에 대한 API 호출을 **개별적으로 모두 생성**해야 합니다.\n"
            "2. **파라미터 채우기**: '필수 파라미터 정보'와 '이전 데이터'를 보고 각 API 호출에 필요한 파라미터 값을 정확히 채워넣으세요.\n"
            "3. **ID 생성**: 각 `step_id`는 `{원래 step_id}-instance-{n}` 형식으로 고유하게 만드세요. (예: '3-1-instance-1')\n\n"
            "--- 입력 정보 ---\n"
            f"### 1. 전체 사용자 질문: {question}\n"
            f"### 2. 이전까지의 API 실행 결과 (JSON): {json.dumps(api_results, ensure_ascii=False, indent=2)}\n"
            f"### 3. 이번에 실행할 스테이지의 엔드포인트 계획 (JSON): {json.dumps(current_stage_plan, ensure_ascii=False, indent=2)}\n\n"
            "--- 출력 (Your turn) ---"
        )
        
        executable_stage_obj = await self.param_planner_model.ainvoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": [self.debug_handler]}
        )
        executable_stage_dict = executable_stage_obj.model_dump()
        print("\n✨ [Dynamically Generated Executable Stage]")
        for step in executable_stage_dict.get('steps', []):
            print(f"   - Step {step['step_id']}: {step['endpoint']} | Params: {step['params']}")
        return {"executable_stage": executable_stage_dict, "current_stage_index": state["current_stage_index"] + 1}

    async def _execute_single_step(self, step_info: Dict[str, Any]):
        result_data = await call_torus_api(step_info['endpoint'], step_info.get('params', {}))
        return {f"step_{step_info['step_id']}": {"request_endpoint": step_info['endpoint'], "request_params": step_info.get('params', {}), "response_data": result_data}}

    async def executor_node(self, state: AgentState):
        print_state_debug("Executor", state)
        steps_to_run = state.get("executable_stage", {}).get('steps', [])
        if not steps_to_run:
            print("   -> No steps to execute.")
            return {}
        
        api_results = state.get("api_results", {})
        print(f"\n🚀 [Executing Stage] - {len(steps_to_run)} parallel API calls")
        tasks = [self._execute_single_step(step) for step in steps_to_run]
        stage_results_list = await asyncio.gather(*tasks)
        
        for res_dict in stage_results_list:
            api_results.update(res_dict)
        return {"api_results": api_results}

    async def synthesizer_node(self, state: AgentState):
        print_state_debug("Synthesizer", state)
        final_prompt = (
            "수집된 API 데이터를 바탕으로 사용자에게 최종 답변을 한국어로 작성하세요.\n\n"
            f"질문: {state['question']}\n"
            f"데이터: {json.dumps(state['api_results'], ensure_ascii=False, indent=2)}"
        )
        response = await self.model.ainvoke(
            [HumanMessage(content=final_prompt)],
            config={"callbacks": [self.debug_handler]}
        )
        return {"final_answer": response.content}
    
    async def save_to_db_node(self, state: AgentState):
        print_state_debug("Save to DB", state)
        if state.get("from_db"):
            print("Skipping DB save because the plan was loaded from DB.")
            return {}
        
        question = state["question"]
        endpoint_plan = state["endpoint_plan"]
        
        db[question] = endpoint_plan
        db_permanent[question] = endpoint_plan
        save_db(DB_PATH, db)
        save_db(DB_PERMANENT_PATH, db_permanent)

        question_embedding = np.array([embedding_model.embed_query(question)], dtype='float32')
        index.add(question_embedding)
        save_faiss_index(index)
            
        print(f"💾 Saved question and plan to DB. Index size: {index.ntotal}")
        return {}

    # --- 조건부 엣지 ---
    
    def decide_branch_after_rag(self, state: AgentState):
        if state.get("from_db"):
            print("--> RAG Success. Jumping to Param Planner.")
            return "param_planner"
        else:
            print("--> RAG Failed. Proceeding to Category Selection.")
            return "category_selector"

    def should_continue_planning(self, state: AgentState):
        if not state.get("executable_stage", {}).get("steps"):
            print("--> All stages complete. Proceeding to Synthesizer.")
            return "synthesizer"
        else:
            print("--> Steps generated. Proceeding to Executor.")
            return "executor"

    # --- 그래프 빌드 ---
    def build_graph(self):
        workflow = StateGraph(AgentState)
        
        workflow.add_node("vector_store", self.vector_store_node)
        workflow.add_node("category_selector", self.category_selection_node)
        workflow.add_node("endpoint_planner", self.endpoint_planner_node)
        workflow.add_node("param_planner", self.param_planner_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        workflow.add_node("save_to_db", self.save_to_db_node)
        
        workflow.set_entry_point("vector_store")
        
        workflow.add_conditional_edges(
            "vector_store",
            self.decide_branch_after_rag,
            {"param_planner": "param_planner", "category_selector": "category_selector"}
        )
        workflow.add_edge("category_selector", "endpoint_planner")
        workflow.add_edge("endpoint_planner", "param_planner")
        
        workflow.add_conditional_edges(
            "param_planner",
            self.should_continue_planning,
            {"executor": "executor", "synthesizer": "synthesizer"}
        )
        workflow.add_edge("executor", "param_planner")
        workflow.add_edge("synthesizer", "save_to_db")
        workflow.add_edge("save_to_db", END)
        
        return workflow.compile()

# -------------------------------------------------------------------------
# 4. Main 실행
# -------------------------------------------------------------------------
async def main():
    # llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-0")
    llm = ChatAnthropic(temperature=0, model="claude-haiku-4-5")

    bot = TorusDynamicAgent(model=llm)
    graph = bot.build_graph()

    # RAG 실패를 유도할 수 있는 복합적인 질문
    query = "지멘스에 탑재된 nc의 모델명이 뭐야"
    
    print(f"User Query: {query}")
    print("="*60)

    async for event in graph.astream({"question": query}):
        for node_name, values in event.items():
            if node_name == "synthesizer" and "final_answer" in values:
                print(f"\n✅ [Final Answer]: {values['final_answer']}")

    print("\n" + "="*60)
    print("✅ 워크플로우 종료")

if __name__ == "__main__":
    asyncio.run(main())
