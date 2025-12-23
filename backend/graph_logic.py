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


# 환경 변수 로드
load_dotenv()
# 기본 로깅은 끄고 커스텀 출력만 봅니다
logging.basicConfig(level=logging.CRITICAL)

# -------------------------------------------------------------------------
# 0. VectorDB 관련 설정
# -------------------------------------------------------------------------
DB_PATH = Path("data/faiss_db.json")
DB_PERMANENT_PATH = Path("data/faiss_db_permanent.json")
INDEX_PATH = Path("data/faiss_index.bin")
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
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    return faiss.IndexFlatL2(768)

def save_faiss_index(index):
    faiss.write_index(index, str(INDEX_PATH))

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

        
        
        # try:
        #     generation = response.generations[0][0]
        #     usage = {}
        #     if hasattr(generation, 'message') and hasattr(generation.message, 'response_metadata'):
        #         usage = generation.message.response_metadata.get('token_usage', {}) or generation.message.response_metadata.get('usage', {})
        #     elif hasattr(response, 'llm_output') and response.llm_output:
        #         usage = response.llm_output.get('token_usage', {})

        #     if usage:
        #         input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
        #         output_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
        #         total = usage.get('total_tokens', 0) or (input_tokens + output_tokens)
        #         print(f"📊 [TOKEN USAGE] Input: {input_tokens} | Output: {output_tokens} | Total: {total}")
        #     else:
        #         print("📊 [TOKEN USAGE] 메타데이터 없음")
        # except Exception as e:
        #     print(f"📊 [TOKEN USAGE] 파싱 실패: {e}")
        if not self.show_token:
            return
        
        try:
            generation = response.generations[0][0]
            usage = {}
            if hasattr(generation, 'message') and hasattr(generation.message, 'usage_metadata'):
                usage = generation.message.usage_metadata

            if usage:

                print(f"  Input tokens: {usage.get('input_tokens', 0)}")
                print(f"  Cache creation: {usage.get('input_token_details', {}).get('cache_creation', 0)}")
                print(f"  Cache read: {usage.get('input_token_details', {}).get('cache_read', 0)}")
                print(f"  Output tokens: {usage.get('output_tokens', 0)}")

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

JSON_PATH = Path(__file__).parent / 'manual' / 'uri_params.json'
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
    def __init__(self, model_sonnet, model_haiku):
        self.debug_handler = DebugCallbackHandler()
        self.model_haiku = model_haiku
        self.model_sonnet = model_sonnet
        self.category_selector_model = model_haiku.with_structured_output(CategorySelection)
        self.endpoint_planner_model = model_haiku.with_structured_output(EndpointExecutionPlan)
        self.param_planner_model = model_sonnet.with_structured_output(ExecutableStage)
        self.PARAMS_JSON = PARAMS_JSON

    async def get_params_info(self, endpoint_list: List[str]):
        results = {}
        for endpoint in endpoint_list:
            endpoint_info = self.PARAMS_JSON.get(endpoint)
            results[endpoint] = endpoint_info.get("required_params") if endpoint_info else "존재하지 않는 엔드포인트"
        return results
    
    # --- 노드 정의 --- || RAG 기능은 없애는 게 좋을 것 같음.
    
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
            f"### 전체 카테고리 목록 (JSON 형식):\n{SHORT_MANUAL}\n\n"
            "분석 후, 관련된 카테고리들의 `category` 필드 값만 JSON 리스트 형태로 반환해주세요."
        )
        messages = [
            {
                "role" : "system",
                "content" : [
                    {
                        "type" : "text",
                        "text" : prompt,
                        "cache_control" : {"type" : "ephemeral"}
                    }
                ]
            },
            {
                "role" : "user",
                "content" : f"### 사용자 질문:\n{question}\n"
            }
        ]
        
        selection_obj = await self.category_selector_model.ainvoke(
            messages,
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
            "2. **Sequential Stage**: 이전 단계의 데이터가 반드시 필요한 경우에만 다음 스테이지로 분리하세요.\n"
            "3. **no duplication**: 엔드포인트는 중복없이 생성합니다. \n\n"
            f"[사용 가능한 API 명세서]\n{documents}"
        )
        messages = [
            {
                "role" : "system",
                "content" : [
                    {
                        "type" : "text",
                        "text" : system_text,
                        "cache_control" : {"type" : "ephemeral"} # 대개 카테고리 별 메뉴얼은 4096토큰 미만이라 haiku 4-5의 경우 캐싱이 안 되지만, sonnet4-5 (캐싱 조건 : 1024토큰)으로 모델을 바꿀 수도 있으므로
                    }
                ]

            },
            {
                "role" : "user",
                "content" : f"### 사용자 질문:\n{question}\n"
            }
        ]

        plan_obj = await self.endpoint_planner_model.ainvoke(
            messages,
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
        response = await self.model_haiku.ainvoke(
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

# API 서버에서 가져다 쓸 수 있도록, LLM과 Agent, Graph(chain)를 모듈 레벨에서 초기화합니다.
llm_haiku = ChatAnthropic(temperature=0, model="claude-haiku-4-5")
llm_sonnet = ChatAnthropic(temperature=0, model="claude-sonnet-4-5")
bot = TorusDynamicAgent(model_haiku=llm_haiku, model_sonnet=llm_sonnet)
chain = bot.build_graph()


# 아래 코드는 'python backend/graph_logic.py'로 직접 실행하여 테스트할 때만 사용됩니다.
async def main():
    # RAG 실패를 유도할 수 있는 복합적인 질문
    query = "화낙에 등록된 공구의 이름을 5개만 알려줘"
    
    print(f"User Query: {query}")
    print("="*60)

    # 이제 'chain' 변수를 직접 사용합니다.
    async for event in chain.astream({"question": query}):
        for node_name, values in event.items():
            if node_name == "synthesizer" and "final_answer" in values:
                print(f"\n✅ [Final Answer]: {values['final_answer']}")

    print("\n" + "="*60)
    print("✅ 워크플로우 종료")

if __name__ == "__main__":
    asyncio.run(main())


CATEGORY_DICT = {
    "장비 상태 및 기본 정보" :
    """
    endpoint 형식:
        • 일반 정보: /machine/{leaf_node}
        • NC 메모리 정보: /machine/ncMemory/{leaf_node}

        === 일반 장비 정보 ===
        • cncModel - 해당 장비에 탑재된 NC의 모델명(STRING)
        • numberOfChannels - 장비에서 사용 가능한 채널(계통)의 개수(INTEGER)
        • cncVendor - NC 제조사 코드 (1: FANUC, 2: SIEMENS, 3:CSCAM, 4: MITSUBISHI, 5: KCNC)(INTEGER)
        • ncLinkState - NC와의 통신 가능 여부(BOOLEAN)
        • currentAccessLevel - 프로그램/디렉토리 접근 권한 수준 (SIEMENS 전용). 1: 제조자, 2: 서비스, 3: 사용자, 4: 프로그래머(키 스위치 3), 5: 공인 전문가(키 스위치 2), 6: 숙련된 전문가(키 스위치 1), 7: 준 숙련 전문가(키 스위치 0)(INTEGER)
        • basicLengthUnit - 장비가 사용하는 기본 길이 단위 (0: Metric, 1: Inches, 4: user Define 등)(INTEGER)
        • machinePowerOnTime - 장비의 전원이 켜진 시간 (단위: 분)(REAL)
        • currentCncTime - 장비에 설정된 현재 시각 (형식: yyyy-MM-ddTHH:mm:ss)(STRING)
        • machineType - 장비의 타입 (0: 알 수 없음, 1: Milling, 2: Lathe)(INTEGER)

        === NC 메모리 정보 ===
        • ncMemory/totalCapacity - NC 메모리의 전체 용량 (단위: byte)(REAL)
        • ncMemory/usedCapacity - 사용 중인 NC 메모리 용량 (단위: byte)(REAL)
        • ncMemory/freeCapacity - NC 메모리의 남은 용량 (단위: byte)(REAL)
        • ncMemory/rootPath - NC 메모리의 기본(루트) 경로(STRING)

        예시:
        - endpoint="/machine/cncModel"
        - endpoint="/machine/ncMemory/freeCapacity"
        - params= {"machine": 1}
    """
    ,
    "계통 별 기록되는 채널의 상태 정보" : 
    """
    endpoint 형식: /machine/channel/{leaf_node}
    
        사용 가능한 leaf_node:
        • channelEnabled  - 해당 채널의 사용 가능 여부(BOOLEAN)
        • toolAreaNumber  - 해당 채널에서 사용 가능한 공구 영역의 식별 번호. 단계통 장비의 경우 디폴트로 1. FANUC에서는 공구 영역과 계통이 동일하기 때문에 channel과 toolArea가 같은 개념으로 사용. SIEMENS의 공구 영역의 개수는 계통 수와 동등하며, 공구 영역과 계통 간 1:다 관계가 성립.(INTEGER)  
        • numberOfAxes  - 해당 채널에서 사용 가능한 축의 개수(INTEGER)
        • numberOfSpindles  - 해당 채널에서 사용 가능한 스핀들의 개수.(INTEGER)
        • alarmStatus   - 채널의 알람 상태. 0: no alarm, 1: alarm, 2: alarm without stop, 3: alarm with stop, 4: Battery low, 5: FAN, 6: PS warning, 7: FSSB waring, 8: Insulate warning, 9: Encoder warning 10: PMC alarm(INTEGER)
        • numberOfAlarms   - 해당 채널에서 발생한 알람의 총 개수(INTEGER)  
        • operateMode   - 공작기계의 운전 모드 (0: JOG, 1: MDI, 2: MEMORY(AUTO), 3: ZRN, 4: MPG, 5: **** 6: EDIT, 7: HANDLE, 8: Teach in JOG, 9: Teach in HANDLE 10: INC·feed, 11: REFERENCE, 12: REMOTE, 13: JOG-REPOS, 14: MDI-REF.POINT, 15: MDI-TEACH IN, 16: MDI-TECH IN-REF.POINT, 17: AUTO-TECH IN-REF.POINT 18: STEP, 19:RAPID, 20: TAPE, 21: AUTO-TEACH IN-JOG, 22: JOG-REF)(INTEGER)
        • numberOfWorkOffsets   - 사용 가능한 공작물 좌표계의 개수(INTEGER)
        • ncState   - CNC의 작동 상태 (0: Reset, 1: Stop, 2: Hold, 3: Start, 4: MSTR, 5: Interrupted, 6: Pause)(INTEGER)
        • motionStatus   - 장비의 현재 모션 상태 (1: Motion, 2: Dwell, 3: Wait)(INTEGER)  
        • emergencyStatus   - 상태 여부 (0: Not emergency, 1: Emergency, 2: Reset, 3: Wait)(INTEGER)
   

        예시: endpoint="/machine/channel/channelEnabled", params={"machine": 1, "channel": 1}
    """
    ,
    "축 별 상태 정보" : 
    """
    endpoint 형식: 
        • 일반 정보: /machine/channel/axis/{leaf_node}
        • 전력 정보: /machine/channel/axis/axisPower/{leaf_node}
            
         === 일반 축 정보 leaf_node ===
        • machinePosition - 기계 좌표계 기준 현재 위치(REAL)
        • workPosition - 공작물 좌표계 기준 현재 위치(REAL)
        • distanceToGo - 지령 위치까지 남은 이동 거리(REAL)
        • relativePosition - 상대 좌표계 기준 현재 위치(REAL)
        • axisName - 절대 좌표계의 축 이름(STRING)
        • relativeAxisName - 상대 좌표계의 축 이름 (FANUC 전용)(STRING)
        • axisLoad - 축에 걸리는 부하(REAL)
        • axisFeed - 현재 축의 이송 속도(REAL)
        • axisLimitPlus - '+' 방향 최대 이동 한계값(REAL)
        • axisLimitMinus - '-' 방향 최대 이동 한계값(REAL)
        • workAreaLimitPlus - 작업 금지 영역 '+' 방향 한계값(REAL)
        • workAreaLimitMinus - 작업 금지 영역 '-' 방향 한계값(REAL)
        • workAreaLimitPlusEnabled - 작업 금지 영역 '+' 방향 활성화 여부(BOOLEAN)
        • workAreaLimitMinusEnabled - 작업 금지 영역 '-' 방향 활성화 여부(BOOLEAN)
        • axisEnabled - 해당 축의 사용 가능 여부(BOOLEAN)
        • interlockEnabled - 해당 축의 인터락 상태 여부(BOOLEAN)
        • constantSurfaceSpeedControlEnabled - 주속 일정 제어(CSS) 활성화 여부(BOOLEAN)
        • axisCurrent - 해당 축의 전류 정보(REAL)
        • machineOrigin - 기계 원점 좌표값(REAL)
        • axisTemperature - 해당 축의 온도 정보(REAL)
        
        === 축 전력 정보 ===  
        • axisPower/actualPowerConsumption - 실 소비 전력 적산값(REAL)
        • axisPower/powerConsumption - 소비 전력 적산값(REAL)
        • axisPower/regeneratedPower - 회생 전력 적산값(REAL)
    
        예시: endpoint="/machine/channel/axis/axisLoad", params={"machine": 1, "channel": 1, "axis": 1}
    """,
    "스핀들 별 상태 정보" : 
    """
    endpoint 형식:
        • 일반 정보: /machine/channel/spindle/{leaf_node}
        • RPM 정보: /machine/channel/spindle/rpm/{leaf_node}
        • 전력 정보: /machine/channel/spindle/spindlePower/{leaf_node}

        === 일반 스핀들 정보 ===
        • spindleLoad - 스핀들에 걸리는 부하(REAL)
        • spindleOverride - 스핀들 속도 오버라이드 비율(REAL)
        • spindleLimit - 최대 회전 속도 한계값(REAL)
        • spindleEnabled - 해당 스핀들의 사용 가능 여부(BOOLEAN)
        • spindleCurrent - 해당 스핀들의 전류 정보(REAL)
        • spindleTemperature - 해당 스핀들의 온도 정보(REAL)

        === 스핀들 RPM 정보 ===
        • rpm/commandedSpeed - 지령된 스핀들 회전 속도(REAL)
        • rpm/actualSpeed - 실제 측정된 스핀들 회전 속도(REAL)
        • rpm/speedUnit - 속도 단위 (0: mm/min, 1: inch/min, 2: rpm, 3: mm/rev, 4: inch/rev 등)(INTEGER)

        === 스핀들 전력 정보 ===
        • spindlePower/actualPowerConsumption - 실 소비 전력의 적산값(REAL)
        • spindlePower/powerConsumption - 소비 전력의 적산값(REAL)
        • spindlePower/regeneratedPower - 회생 전력의 적산값(REAL)

        예시:
        - endpoint="/machine/channel/spindle/spindleLoad"
        - endpoint="/machine/channel/spindle/rpm/actualSpeed"
        - endpoint="/machine/channel/spindle/spindlePower/powerConsumption"
        - params={"machine": 1, "channel": 1, "spindle": 1}
    """,
    "축 이송 정보" : 
    """
    endpoint 형식:
        • 오버라이드 정보: /machine/channel/feed/{leaf_node}
        • 이송 속도 정보: /machine/channel/feed/feedRate/{leaf_node}

        === 이송 오버라이드 정보 ===
        • feedOverride - 가공 이송 속도 오버라이드 비율(REAL)
        • rapidOverride - 급속 이송 속도 오버라이드 비율(REAL)

        === 이송 속도 정보 ===
        • feedRate/commandedSpeed - 지령된 이송 속도(REAL)
        • feedRate/actualSpeed - 실제 측정된 이송 속도(REAL)
        • feedRate/speedUnit - 속도 단위  (0: mm/min, 1: inch/min, 2: rpm, 3: mm/rev, 4: inch/rev 등)(INTEGER)

        예시:
        - endpoint="/machine/channel/feed/feedOverride"
        - endpoint="/machine/channel/feed/feedRate/actualSpeed"
        - params={"machine": 1, "channel": 1}
    """,
    "가공 작업의 진척 상태 정보" : 
    """
    endpoint 형식:
        • 가공 수량 정보: /machine/channel/workStatus/workCounter/{leaf_node}
        • 가공 시간 정보: /machine/channel/workStatus/machiningTime/{leaf_node}


        === 가공 수량 정보 ===
        • workCounter/currentWorkCounter - 현재까지 가공한 수량(INTEGER)
        • workCounter/targetWorkCounter - 목표 가공 수량(INTEGER)
        • workCounter/totalWorkCounter - 총 가공 수량(INTEGER)

        === 가공 시간 정보 ===
        • machiningTime/processingMachiningTime - 현재 가공이 진행된 시간 (단위: 초)(REAL)
        • machiningTime/estimatedMachiningTime - 예상 남은 가공 완료 시간 (SIEMENS 전용)(REAL)
        • machiningTime/machineOperationTime - 자동 운전 모드에서의 총 운전 시간 (단위: 초)(REAL)
        • machiningTime/actualCuttingTime - 실제 총 절삭 시간 (단위: 초)(REAL)

        예시:
        - endpoint="/machine/channel/workStatus/workCounter/currentWorkCounter"
        - params={"machine": 1, "channel": 1, "workStatus": 1}
        - endpoint="/machine/channel/workStatus/machiningTime/processingMachiningTime"
        - params={"machine": 1, "channel": 1, "workStatus": 1}
    """,
    "현재 활성된 공구의 상세 정보" :  
    """
    endpoint 형식:
        • 일반 정보: /machine/channel/activeTool/{leaf_node}
        • 공구 날 정보: /machine/channel/activeTool/toolEdge/{leaf_node}
        • 공구 수명 정보: /machine/channel/activeTool/toolEdge/toolLife/{leaf_node}

        === 일반 공구 정보 ===
        • locationNumber - 공구가 매거진에 탑재된 위치 번호(INTEGER)
        • toolName - 공구 이름(STRING)
        • toolNumber - 공구 식별 번호 (T 코드)(INTEGER)
        • numberOfEdges - 공구 날의 총 개수(INTEGER)
        • toolEnabled - 공구 영역 등록 및 매거진 탑재 여부 0: 공구 영역 미등록, 매거진 미탑재 상태, 1: 공구 영역 등록, 매거진 미탑재 상태, 2: 공구 영역 등록, 매거진 탑재 상태(INTEGER)
        • magazineNumber - 공구가 탑재된 매거진 번호(INTEGER)
        • sisterToolNumber - 할당된 대체 공구 번호(INTEGER)
        • toolLifeUnit - 공구 수명 측정 단위 기준.  0: no unit, 1: time, 2: count, 4: wear, 5: count(장착) 6: count(사용), 8: offset (INTEGER)
        • toolGroupNumber - 공구가 참조된 공구 그룹 번호 리스트(INTEGER)
        • toolUseOrderNumber - 그룹 내 공구 사용 순서 (FANUC 전용)(INTEGER)
        • toolStatus - 공구의 사용 상태 0 : Not enabled, 1 : Active tool, 2 : Enabled, 4 : Disabled, 8 : Measured, 9: 미사용 공구, 10 : 정상 수명 공구, 11 : Tool data is available (using), 12 : This tool is registered (available), 13 : This tool has expired, 14 : This tool was skipped, 16 : Prewarning limit reached , 32 : Tool being changed , 64 : Fixed location coded, 128 : Tool was in use , 256 : Tool is in the buffer magazine with transport order, 512 : Ignore disabled state of tool, 1024 : Tool must be unloaded, 2048 : Tool must be loaded, 4096 : Tool is a master tool, 8192 : Reserved, 16384 : Tool is marked for 1:1 exchange, 32768 : Tool is being used as a manual tool (INTEGER)

        === 공구 날(Edge) 정보 ===
        • toolEdge/edgeNumber - 공구 날 식별 번호(INTEGER)
        • toolEdge/toolType - 공구 유형 0: Not defined, 10: General-purpose tool, 11: Threading tool (Siemens에서는 540), 12: Grooving tool, 13: Round-nose tool, 14: Point nose straight tool, 15: Versatile tool, 20: Drill, 21: Counter sink tool, 22: Flat end mill, 23: Ball end mill, 24: Tap (Siemens에서는 240), 25: Reamer, 26: Boring tool, 27: Face mill, 50: Radius end mill, 51: 면취, 52: 선삭, 53: 홈삽입, 54: 나사절삭, 55: 선삭드릴, 56: 선삭탭, 100: Milling tool, 110: Ball nose end mill, 111: Conical ball end, 120: End mill, 121: End mill corner rounding, 130: Angle head cutter, 131: Corner rounding angle head cutter, 140: Facing tool, 145: Thread cutter, 150: Side mill, 151: Saw, 155: Bevelled cutter, 156: Bevelled cutter corner, 157: Tap. die-sink. cutter, 160: Drill&thread cut., 200: Twist drill, 205: Solid drill, 210: Boring bar, 220: Center drill, 230: Countersink, 231: Counterbore, 240: Tap, 241: Fine tap, 242: Tap, Whitworth, 250: Reamer, 500: Roughing tool, 510: Finishing tool, 520: Plunge cutter, 530: Cutting tool, 540: Threading tool, 550: Button tool, 560: Rotary drill, 580: 3D turning probe, 585: Calibrating tool, 700: Slotting saw, 710: 3D probe, 711: Edge finder, 712: Mono probe, 713: L probe, 714: Star probe, 725: Calibrating tool, 730: Stop, 731: Mandrel, 732: Steady rest, 900: Auxiliary tools(INTEGER)
        • toolEdge/lengthOffsetNumber - 공구 길이 보정 식별 번호(INTEGER)
        • toolEdge/geoLengthOffset - 공구 길이 X 보정값(REAL)
        • toolEdge/wearLengthOffset - 공구 길이 X 마모 보정값(REAL)
        • toolEdge/radiusOffsetNumber - 공구 반경 보정 식별 번호(INTEGER)
        • toolEdge/geoRadiusOffset - 공구 반경 보정값(REAL)
        • toolEdge/wearRadiusOffset - 공구 반경 마모 보정값(REAL)
        • toolEdge/edgeEnabled - 공구 날 사용 가능 여부(BOOLEAN)
        • toolEdge/geoLengthOffsetZ - 공구 길이 Z 보정값(REAL)
        • toolEdge/wearLengthOffsetZ - 공구 길이 Z 마모 보정값(REAL)
        • toolEdge/geoLengthOffsetY - 공구 길이 Y 보정값(REAL)
        • toolEdge/wearLengthOffsetY - 공구 길이 Y 마모 보정값(REAL)
        • toolEdge/geoOffsetNumber - 길이 X,Z, 반경의 식별 번호(INTEGER)
        • toolEdge/wearOffsetNumber - 길이 X,Z, 반경 마모값의 식별 번호(INTEGER)
        • toolEdge/cuttingEdgePosition - 공구 인선 방향(INTEGER)
        • toolEdge/tipAngle - 공구의 팁 각도(REAL)
        • toolEdge/holderAngle - 공구 홀더 각도(REAL)
        • toolEdge/insertAngle - 공구 인서트 각도(REAL)
        • toolEdge/insertWidth - 인선 너비 (SIEMENS 전용)(REAL)
        • toolEdge/insertLength - 인선 길이 (SIEMENS 전용)(REAL)
        • toolEdge/referenceDirectionHolderAngle - 홀더 각도 참조 방향 (SIEMENS 전용)(REAL)
        • toolEdge/directionOfSpindleRotation - 스핀들 회전 방향  0: 회전 없음, 1: 시계 방향, 2: 반시계 방향(SIEMENS 전용)(INTEGER)
        • toolEdge/numberOfTeeth - 공구 날 개수 (SIEMENS 전용)(INTEGER)
        
        === 공구 수명 정보 ===
        • toolEdge/toolLife/maxToolLife - 최대 공구 수명(REAL)
        • toolEdge/toolLife/restToolLife - 잔여 공구 수명(REAL)
        • toolEdge/toolLife/toolLifeCount - 현재 공구 사용량(REAL)
        • toolEdge/toolLife/toolLifeAlarm - 공구 수명 도달 경고 설정값 (SIEMENS 전용)(REAL)

        예시:
        - params = {"machine": 1, "channel": 1}
        - endpoint = "/machine/channel/activeTool/toolNumber"
        
        - params = {"machine": 1, "channel": 1}
        - endpoint = "/machine/channel/activeTool/toolEdge/geoLengthOffset"

        - params = {"machine": 1, "channel": 1}
        - endpoint = "/machine/channel/activeTool/toolEdge/toolLife/restToolLife"
    """,
    "현재 실행 중인 NC 프로그램의 상태 정보" : 
    """
    endpoint 형식:
        • 일반 정보: /machine/channel/currentProgram/{leaf_node}
        • 모달 정보: /machine/channel/currentProgram/modal/{leaf_node}
        • 실행 블록 정보: /machine/channel/currentProgram/overallBlock/{leaf_node}
        • 중단점 정보: /machine/channel/currentProgram/interruptBlock/{leaf_node}
        • 좌표계 오프셋 정보: /machine/channel/currentProgram/currentTotalWorkOffset/{leaf_node}
        • 현재 파일 정보: /machine/channel/currentProgram/currentFile/{leaf_node}
        • 메인 파일 정보: /machine/channel/currentProgram/mainFile/{leaf_node}
        • 제어 옵션 정보: /machine/channel/currentProgram/controlOption/{leaf_node}


        === 일반 프로그램 정보 ===
        • sequenceNumber - 현재 실행 중인 시퀀스 번호(N 코드)(INTEGER)
        • currentBlockCounter - 실행 중인 블록 카운터(INTEGER)
        • lastBlock - 이전 블록 정보(STRING)
        • currentBlock - 현재 실행 중인 프로그램 블록 내용(STRING)
        • nextBlock - 다음 블록 정보(STRING)
        • activePartProgram - 실행 중인 프로그램 블록 정보(최대 200자)(STRING)
        • programMode - 프로그램 실행 모드  0: Reset, 1: Stop, 2: Hold, 3: Start(Active)(run), 4: MSTR, 5: Interrupted, 6: Pause, 7: Waiting (INTEGER)
        • currentWorkOffsetIndex - 현재 공작물 좌표계의 G 코드 인덱스(INTEGER)
        • currentWorkOffsetCode - 현재 공작물 좌표계의 G 코드 문자열(STRING)
        • currentDepthLevel - 현재 프로그램의 레벨 (메인, 서브루틴 등)(INTEGER)

        === G 코드 모달 정보 ===
        • modal/modalIndex - G 코드 인덱스(INTEGER)
        • modal/modalCode - G 코드 문자열(STRING)

        === 실행 블록 정보 (SIEMENS) ===
        • overallBlock/blockCounter - 블록 카운터(INTEGER)
        • overallBlock/programName - 프로그램 이름(STRING)

        === 중단점 블록 정보 (SIEMENS) ===
        • interruptBlock/depthLevel - 중단점 블록의 프로그램 레벨 (INTEGER)
        • interruptBlock/blockCounter - 중단점 블록의 카운터(INTEGER)
        • interruptBlock/programName - 중단점 블록의 프로그램 이름 (STRING)
        • interruptBlock/blockData - 중단점 블록 데이터 (STRING)
        • interruptBlock/searchType - 중단점 검색 유형 (INTEGER)
        • interruptBlock/mainProgramName - 중단점의 메인 프로그램 이름 (STRING)

        === 공작물 좌표계 오프셋 정보 ===
        • currentTotalWorkOffset/workOffsetIndex - G 코드 인덱스(INTEGER)
        • currentTotalWorkOffset/workOffsetValue - 축별 총 오프셋 값 (REAL)
        • currentTotalWorkOffset/workOffsetRotation - 축별 총 회전 오프셋 값 (REAL)
        • currentTotalWorkOffset/workOffsetScalingFactor - 축별 총 스케일링 값 (REAL)
        • currentTotalWorkOffset/workOffsetMirroringEnabled - 축별 미러링 활성화 여부 (BOOLEAN)

        === 현재 실행 파일 정보 ===
        • currentFile/programName - 파일명(STRING)
        • currentFile/programPath - 파일 경로(STRING)
        • currentFile/programSize - 파일 크기 (byte)(REAL)
        • currentFile/programDate - 파일 생성 날짜(STRING)
        • currentFile/programNameWithPath - 경로를 포함한 전체 파일명(STRING)

        === 메인 프로그램 파일 정보 ===
        • mainFile/programName - 파일명(STRING)
        • mainFile/programPath - 파일 경로(STRING)
        • mainFile/programSize - 파일 크기 (byte)(REAL)
        • mainFile/programDate - 파일 생성 날짜(STRING)
        • mainFile/programNameWithPath - 경로를 포함한 전체 파일명(STRING)
        
        === 프로그램 제어 옵션 정보 ===
        • controlOption/singleBlock - 싱글 블록 실행 여부(BOOLEAN)
        • controlOption/dryRun - 드라이 런 실행 여부(BOOLEAN)
        • controlOption/optionalStop - 옵셔널 스톱(M01) 활성화 여부(BOOLEAN)
        • controlOption/blockSkip - 블록 스킵 활성화 여부 리스트 (BOOLEAN)
        • controlOption/machineLock - 머신 락 활성화 여부(BOOLEAN)

        예시:
        - params = {"machine": 1, "channel": 1}
        - endpoint = "/machine/channel/currentProgram/sequenceNumber"

        - params = {"machine": 1, "channel": 1, "modalCode": 1}
        - endpoint = "/machine/channel/currentProgram/modal/modalCode"

        - params = {"machine": 1, "channel": 1, "workOffsetValue": 1}
        - endpoint = "/machine/channel/currentProgram/currentTotalWorkOffset/workOffsetValue"
        
        - params = {"machine": 1, "channel": 1, "blockSkip": 1}
        - endpoint = "/machine/channel/currentProgram/controlOption/blockSkip"
    """,
    "공작물 좌표계의 오프셋 정보" : 
    """
    endpoint 형식:
        • 오프셋 정보: /machine/channel/workOffset/{leaf_node}

        필수 파라미터: machine=i, channel=j, workOffset=k 와 아래 각 항목별 파라미터

        === 공작물 좌표계 오프셋 정보 ===
        • workOffsetValue - G 코드 인덱스에 대한 축별 오프셋 값 (REAL)
        • workOffsetRotation - 축별 오프셋 회전량 (SIEMENS 전용)(REAL)
        • workOffsetScalingFactor - 축별 오프셋 확장량 (SIEMENS 전용) (REAL)
        • workOffsetMirroringEnabled - 축별 미러링 활성화 여부 (SIEMENS 전용) (BOOLEAN)
        • workOffsetFine - 축별 오프셋 Fine 값 (SIEMENS 전용) (REAL)

        예시:
        # G54(workOffset=1) 좌표계의 1번째 축(workOffsetValue=1) 오프셋 값을 조회
        - params = {"machine": 1, "channel": 1, "workOffset": 1, "workOffsetValue": 1}
        - endpoint = "/machine/channel/workOffset/workOffsetValue"
    """,
    "알람 정보" : 
    """
    endpoint 형식:
        • 알람 정보: /machine/channel/alarm/{leaf_node}

        === 알람 정보 ===
        • (수정하자) - 해당 계통에서 발생한 모든 알람에 대한 Text, Category, Number, raisedTimeStamp를 리스트로 나타내는 문자열(JSON 형태)(INTEGER)
        • alarmText - 알람 상세 내용 (STRING)
        • alarmCategory - 알람 유형 (STRING)
        • alarmNumber - 알람 번호 (STRING)
        • raisedTimeStamp - 알람 발생 시각 (STRING)

        예시:
        # 1번째 발생 알람의 상세 내용을 조회
        - params = {"machine": 1, "channel": 1, "alarm": 1}
        - endpoint = "/machine/channel/alarm/alarmText"
    """,
    "사용자 변수" : 
    """
    • userVariable - 사용자 변수 값(REAL)
        
        endpoint 형식: /machine/channel/variable/{leaf_node}
    """,
    "CNC 내부 PLC 메모리 데이터" : 
    """
    endpoint 형식:
        • 메모리 정보: /machine/pic/memory/{leaf_node}

        === PLC 메모리 정보 ===
        • rbitBlock - 읽기 전용 Bit 데이터 블록 (BOOLEAN)
        • bitBlock - 읽기/쓰기 가능 Bit 데이터 블록 (BOOLEAN)
        • rbyteBlock - 읽기 전용 Byte 데이터 블록 (BYTE)
        • byteBlock - 읽기/쓰기 가능 Byte 데이터 블록 (BYTE)
        • rwordBlock - 읽기 전용 Word(2byte) 데이터 블록 (WORD)
        • wordBlock - 읽기/쓰기 가능 Word(2byte) 데이터 블록 (WORD)
        • rdwordBlock - 읽기 전용 DWord(4byte) 데이터 블록 (DWORD)
        • dwordBlock - 읽기/쓰기 가능 DWord(4byte) 데이터 블록 (DWORD)
        • rqwordBlock - 읽기 전용 QWord(8byte) 데이터 블록 (QWORD)
        • qwordBlock - 읽기/쓰기 가능 QWord(8byte) 데이터 블록 (QWORD)

        예시:
        # 100번 주소의 읽기 전용 Bit 블록 값을 조회
        - params = {"machine": 1, "rbitBlock": 100}
        - endpoint = "/machine/pic/memory/rbitBlock"

        # 200번 주소의 읽기/쓰기 Word 블록 값을 조회
        - params = {"machine": 1, "wordBlock": 200}
        - endpoint = "/machine/pic/memory/wordBlock"
    """,
    "장비 공구 영역 정보" : 
    """
    endpoint 형식:
        • 일반 정보: /machine/toolArea/{leaf_node}
        • 매거진 정보: /machine/toolArea/magazine/{leaf_node}
        • T코드 기준 공구 정보: /machine/toolArea/tools/{leaf_node}
        • T코드 기준 공구 날 정보: /machine/toolArea/tools/toolEdge/{leaf_node}
        • T코드 기준 공구 수명 정보: /machine/toolArea/tools/toolEdge/toolLife/{leaf_node}
        • 등록순 기준 공구 정보: /machine/toolArea/registerTools/{leaf_node}
        • 등록순 기준 공구 날 정보: /machine/toolArea/registerTools/toolEdge/{leaf_node}
        • 등록순 기준 공구 수명 정보: /machine/toolArea/registerTools/toolEdge/toolLife/{leaf_node}

        === 일반 공구 영역 정보 ===
        • toolAreaEnabled - 해당 공구 영역 사용 가능 여부 (BOOLEAN)
        • numberOfMagazines - 사용 가능한 매거진 개수 (INTEGER)
        • numberOfRegisteredTools - 공구 영역에 등록된 총 공구 개수 (INTEGER)
        • numberOfLoadedTools - 매거진에 탑재된 총 공구 개수 (INTEGER)
        • numberOfToolGroups - 등록된 공구 그룹의 개수 (INTEGER)
        • numberOfToolOffsets - 등록된 공구 오프셋의 개수 (INTEGER)

        === 매거진 정보 ===
        • magazine/magazineEnabled - 해당 매거진 사용 가능 여부 (BOOLEAN)
        • magazine/magazineName - 매거진 이름 (SIEMENS 전용) (STRING)
        • magazine/numberOfRealLocations - 매거진의 물리적 포트(위치) 개수 (INTEGER)
        • magazine/magazinePhysicalNumber - 매거진의 물리적 번호 (INTEGER)
        • magazine/numberOfLoadedTools - 해당 매거진에 탑재된 공구 개수 (INTEGER)

        === 공구 상세 정보 ===
        # 아래 항목들은 tools와 registerTools 경로에서 동일하게 사용됩니다. (예: /machine/toolArea/tools/toolName)
        • locationNumber - 공구가 매거진에 탑재된 위치 번호 (INTEGER)
        • toolName - 공구 이름 (STRING)
        • numberOfEdges - 공구 날의 총 개수 (INTEGER)
        • toolEnabled - 공구 영역 등록 및 매거진 탑재 여부 0: 공구 영역 미등록, 매거진 미탑재 상태, 1: 공구 영역 등록, 매거진 미탑재 상태, 2: 공구 영역 등록, 매거진 탑재 상태(INTEGER) 
        • magazineNumber - 공구가 탑재된 매거진 번호 (INTEGER)
        • sisterToolNumber - 할당된 대체 공구 번호 (INTEGER)
        • toolLifeUnit - 공구 수명 측정 단위 기준 (INTEGER)
        • toolGroupNumber - 공구가 참조된 공구 그룹 번호 리스트 (LIST[INTEGER])
        • toolUseOrderNumber - 그룹 내 공구 사용 순서 (FANUC 전용) (INTEGER)
        • toolStatus - 공구의 사용 상태 0 : Not enabled, 1 : Active tool, 2 : Enabled, 4 : Disabled, 8 : Measured, 9: 미사용 공구, 10 : 정상 수명 공구, 11 : Tool data is available (using), 12 : This tool is registered (available), 13 : This tool has expired, 14 : This tool was skipped, 16 : Prewarning limit reached , 32 : Tool being changed , 64 : Fixed location coded, 128 : Tool was in use , 256 : Tool is in the buffer magazine with transport order, 512 : Ignore disabled state of tool, 1024 : Tool must be unloaded, 2048 : Tool must be loaded, 4096 : Tool is a master tool, 8192 : Reserved, 16384 : Tool is marked for 1:1 exchange, 32768 : Tool is being used as a manual tool (INTEGER)

        === 공구 날(Edge) 상세 정보 ===
        # 아래 항목들은 .../tools/toolEdge 및 .../registerTools/toolEdge 경로에서 동일하게 사용됩니다.
        • toolType - 공구 유형 0: Not defined, 10: General-purpose tool, 11: Threading tool (Siemens에서는 540), 12: Grooving tool, 13: Round-nose tool, 14: Point nose straight tool, 15: Versatile tool, 20: Drill, 21: Counter sink tool, 22: Flat end mill, 23: Ball end mill, 24: Tap (Siemens에서는 240), 25: Reamer, 26: Boring tool, 27: Face mill, 50: Radius end mill, 51: 면취, 52: 선삭, 53: 홈삽입, 54: 나사절삭, 55: 선삭드릴, 56: 선삭탭, 100: Milling tool, 110: Ball nose end mill, 111: Conical ball end, 120: End mill, 121: End mill corner rounding, 130: Angle head cutter, 131: Corner rounding angle head cutter, 140: Facing tool, 145: Thread cutter, 150: Side mill, 151: Saw, 155: Bevelled cutter, 156: Bevelled cutter corner, 157: Tap. die-sink. cutter, 160: Drill&thread cut., 200: Twist drill, 205: Solid drill, 210: Boring bar, 220: Center drill, 230: Countersink, 231: Counterbore, 240: Tap, 241: Fine tap, 242: Tap, Whitworth, 250: Reamer, 500: Roughing tool, 510: Finishing tool, 520: Plunge cutter, 530: Cutting tool, 540: Threading tool, 550: Button tool, 560: Rotary drill, 580: 3D turning probe, 585: Calibrating tool, 700: Slotting saw, 710: 3D probe, 711: Edge finder, 712: Mono probe, 713: L probe, 714: Star probe, 725: Calibrating tool, 730: Stop, 731: Mandrel, 732: Steady rest, 900: Auxiliary tools (INTEGER) (INTEGER)
        • lengthOffsetNumber - 공구 길이 보정 식별 번호 (INTEGER)
        • toolEdge/lengthOffsetNumber - 공구 길이 보정 식별 번호 (INTEGER)
        • toolEdge/geoLengthOffset - 공구 길이 X 보정값 (REAL)
        • toolEdge/wearLengthOffset - 공구 길이 X 마모 보정값 (REAL)
        • toolEdge/radiusOffsetNumber - 공구 반경 보정 식별 번호 (INTEGER)
        • toolEdge/geoRadiusOffset - 공구 반경 보정값 (REAL)
        • toolEdge/wearRadiusOffset - 공구 반경 마모 보정값 (REAL)
        • toolEdge/edgeEnabled - 공구 날 사용 가능 여부 (BOOLEAN)
        • toolEdge/geoLengthOffsetZ - 공구 길이 Z 보정값(REAL)
        • toolEdge/wearLengthOffsetZ - 공구 길이 Z 마모 보정값(REAL)
        • toolEdge/geoLengthOffsetY - 공구 길이 Y 보정값 (REAL)
        • toolEdge/wearLengthOffsetY - 공구 길이 Y 마모 보정값 (REAL)
        • toolEdge/geoOffsetNumber - 길이 X,Z, 반경의 식별 번호 (INTEGER)
        • toolEdge/wearOffsetNumber - 길이 X,Z, 반경 마모값의 식별 번호 (INTEGER)
        • toolEdge/cuttingEdgePosition - 공구 인선 방향 (INTEGER)
        • toolEdge/tipAngle - 공구의 팁 각도 (REAL)
        • toolEdge/holderAngle - 공구 홀더 각도 (REAL)
        • toolEdge/insertAngle - 공구 인서트 각도 (REAL)
        • toolEdge/insertWidth - 인선 너비 (SIEMENS 전용) (REAL)
        • toolEdge/insertLength - 인선 길이 (SIEMENS 전용) (REAL)
        • toolEdge/referenceDirectionHolderAngle - 홀더 각도 참조 방향 (SIEMENS 전용) (REAL)
        • toolEdge/directionOfSpindleRotation - 스핀들 회전 방향 (SIEMENS 전용) (INTEGER)
        • toolEdge/numberOfTeeth - 공구 날 개수 (SIEMENS 전용) (INTEGER)

        === 공구 수명 상세 정보 ===
        # 아래 항목들은 .../toolEdge/toolLife 경로에서 동일하게 사용됩니다.
        • toolLife/maxToolLife - 최대 공구 수명 (REAL)
        • toolLife/restToolLife - 잔여 공구 수명 (REAL)
        • toolLife/toolLifeCount - 현재 공구 사용량 (REAL)
        • toolLife/toolLifeAlarm - 공구 수명 도달 경고 설정값 (REAL)

        예시:
        # 1번 공구 영역의 매거진 개수 조회
        - params = {"machine": 1, "toolArea": 1}
        - endpoint = "/machine/toolArea/numberOfMagazines"

        # T코드 5번 공구의 이름 조회
        - params = {"machine": 1, "toolArea": 1, "tools": 5}
        - endpoint = "/machine/toolArea/tools/toolName"

        # T코드 5번, 1번 날(Edge), 1번 그룹의 길이 X 보정값 조회
        - params = {"machine": 1, "toolArea": 1, "tools": 5, "toolEdge": 1, "geoLengthOffset": 1}
        - endpoint = "/machine/toolArea/tools/toolEdge/geoLengthOffset"
        
        # 등록순 3번 공구, 1번 날(Edge), 1번 그룹의 잔여 수명 조회
        - params = {"machine": 1, "toolArea": 1, "registerTools": 3, "toolEdge": 1, "restToolLife": 1}
        - endpoint = "/machine/toolArea/registerTools/toolEdge/toolLife/restToolLife"
    """,
    "내장 센서 데이터의 시계열 수집 정보" : 
    """
    endpoint 형식:
        • 버퍼 정보: /machine/buffer/{leaf_node}
        • 스트림 정보: /machine/buffer/stream/{leaf_node}

        === 버퍼 정보 ===
        • bufferEnabled - 해당 버퍼 사용 가능 여부 (BOOLEAN)
        • numberOfStream - 해당 버퍼의 최대 스트림 개수 (INTEGER)
        • statusOfStream - 스트림 상태 (0: 설정 가능, 1: 수집 가능, 2: 수집 대기, 3: 수집 중, 4: 수집 대기 혹은 수집 중, 5: 수집 완료/종료, -1: CNC 연결 실패, -2: 설정값 적용 실패 등)(필수: buffer=j) (INTEGER)
        • modOfStream - 스트림 수집 모드 (0: 반복 수집, 1: 1회 수집) (INTEGER)
        • machineChannelOfStream - 스트림 수집 시 사용할 채널(INTEGER)
        • periodOfStream - 1회 수집 기간 (단위: ms) (INTEGER)
        • triggerOfStream - 수집 시작 트리거 (0: 즉시, 1이상: 시퀀스 번호)(INTEGER)
        • frequencyOfStream - 모든 스트림에 공통으로 적용할 수집 주파수 (Hz)(INTEGER)

        === 스트림 채널 정보 ===
        • stream/streamEnabled - 해당 스트림 사용 가능 여부 (BOOLEAN)
        • stream/streamFrequency - 해당 스트림의 수집 주파수 (Hz) (INTEGER)
        • stream/streamCategory - 수집 대상 데이터 카테고리 (INTEGER)
        • stream/streamSubcategory - 수집 대상 데이터 서브카테고리 (축/스핀들 번호 등)(INTEGER)
        • stream/streamType - 수집 유형 (KCNC 전용) (INTEGER)
        • stream/streamStartBit - 수집 유형이 Bit일 때 Start Bit (KCNC 전용) (INTEGER)
        • stream/streamEndBit - 수집 유형이 Bit일 때 End Bit (KCNC 전용) (INTEGER)
        • stream/value - 해당 스트림에서 마지막으로 수집된 데이터 값(REAL)

        예시:
        # 1번 버퍼의 수집 상태를 조회
        - params = {"machine": 1, "buffer": 1}
        - endpoint = "/machine/buffer/statusOfStream"

        # 1번 버퍼의 3번 스트림에서 마지막으로 수집된 값을 조회
        - params = {"machine": 1, "buffer": 1, "stream": 3}
        - endpoint = "/machine/buffer/stream/value"
    """
}

SHORT_MANUAL = """

[
  {
    "category": "장비 상태 및 기본 정보",
    "keywords": [
      "cncModel",
      "numberOfChannels",
      "cncVendor",
      "ncLinkState",
      "currentAccessLevel",
      "basicLengthUnit",
      "machinePowerOnTime",
      "currentCncTime",
      "machineType",
      "totalCapacity",
      "usedCapacity",
      "freeCapacity",
      "rootPath"
    ],
    "description": "NC 컨트롤러의 모델명(Fanuc, Siemens 등), 제조사 코드, 통신 연결 상태(Link State), 시스템 시간 등 장비의 정적 제원을 조회합니다. 또한 NC 메모리의 전체/사용/잔여 용량(Byte)과 루트 경로 정보를 포함하여 시스템 리소스 현황을 전반적으로 파악할 수 있습니다."
  },
  {
    "category": "계통 별 기록되는 채널의 상태 정보",
    "keywords": [
      "channelEnabled",
      "toolAreaNumber",
      "numberOfAxes",
      "numberOfSpindles",
      "alarmStatus",
      "numberOfAlarms",
      "operateMode",
      "numberOfWorkOffsets",
      "ncState",
      "motionStatus",
      "emergencyStatus"
    ],
    "description": "각 채널(계통)의 활성화 여부, 알람 발생 상태(No alarm, Stop 등), 현재 운전 모드(JOG, MEMORY, MDI 등)와 NC의 작동 상태(Run, Stop, Hold)를 모니터링합니다. 해당 채널에 구성된 축, 스핀들, 공구 영역의 개수 정보도 포함됩니다."
  },
  {
    "category": "축 별 상태 정보",
    "keywords": [
      "machinePosition",
      "workPosition",
      "distanceToGo",
      "relativePosition",
      "axisName",
      "relativeAxisName",
      "axisLoad",
      "axisFeed",
      "axisLimitPlus",
      "axisLimitMinus",
      "workAreaLimitPlus",
      "workAreaLimitMinus",
      "workAreaLimitPlusEnabled",
      "workAreaLimitMinusEnabled",
      "axisEnabled",
      "interlockEnabled",
      "constantSurfaceSpeedControlEnabled",
      "axisCurrent",
      "machineOrigin",
      "axisTemperature",
      "actualPowerConsumption",
      "powerConsumption",
      "regeneratedPower"
    ],
    "description": "각 축의 실시간 위치 정보(기계, 공작물, 상대 좌표)와 잔여 이동 거리, 축에 걸리는 부하율(Load), 이송 속도, 전류 및 온도 데이터를 모니터링합니다. 소프트웨어 리미트, 작업 금지 영역 설정 상태 및 소비/회생 전력량 데이터도 상세히 제공합니다."
  },
  {
    "category": "스핀들 별 상태 정보",
    "keywords": [
      "spindleLoad",
      "spindleOverride",
      "spindleLimit",
      "spindleEnabled",
      "spindleCurrent",
      "spindleTemperature",
      "commandedSpeed",
      "actualSpeed",
      "speedUnit",
      "actualPowerConsumption",
      "powerConsumption",
      "regeneratedPower"
    ],
    "description": "스핀들의 지령 및 실제 회전 속도(RPM), 속도 오버라이드 비율, 부하율, 전류, 온도 상태를 실시간으로 확인합니다. 또한 스핀들 구동에 따른 실시간 소비 전력과 회생 전력 적산값을 제공하여 에너지 효율 분석을 지원합니다."
  },
  {
    "category": "축 이송 정보",
    "keywords": [
      "feedOverride",
      "rapidOverride",
      "commandedSpeed",
      "actualSpeed",
      "speedUnit"
    ],
    "description": "가공 경로 이동(Feed) 및 급속 이송(Rapid) 시 적용된 오버라이드 비율을 조회합니다. 또한 단위 시간당 이송 속도(Feed Rate)의 지령값과 실제 측정값을 제공하여 가공 속도 제어 상태를 확인합니다."
  },
  {
    "category": "가공 작업의 진척 상태 정보",
    "keywords": [
      "currentWorkCounter",
      "targetWorkCounter",
      "totalWorkCounter",
      "processingMachiningTime",
      "estimatedMachiningTime",
      "machineOperationTime",
      "actualCuttingTime"
    ],
    "description": "현재 진행 중인 작업의 목표 수량 대비 가공 실적(Current/Total Work Counter)을 추적합니다. 사이클 타임, 실제 절삭 시간, 자동 운전 모드 가동 시간 및 예상 잔여 시간(Siemens) 등 생산성 분석을 위한 핵심 시간 데이터를 제공합니다."
  },
  {
    "category": "현재 활성된 공구의 상세 정보",
    "keywords": [
      "locationNumber",
      "toolName",
      "toolNumber",
      "numberOfEdges",
      "toolEnabled",
      "magazineNumber",
      "sisterToolNumber",
      "toolLifeUnit",
      "toolGroupNumber",
      "toolUseOrderNumber",
      "toolStatus",
      "edgeNumber",
      "toolType",
      "lengthOffsetNumber",
      "geoLengthOffset",
      "wearLengthOffset",
      "radiusOffsetNumber",
      "geoRadiusOffset",
      "wearRadiusOffset",
      "edgeEnabled",
      "geoLengthOffsetZ",
      "wearLengthOffsetZ",
      "geoLengthOffsetY",
      "wearLengthOffsetY",
      "geoOffsetNumber",
      "wearOffsetNumber",
      "cuttingEdgePosition",
      "tipAngle",
      "holderAngle",
      "insertAngle",
      "insertWidth",
      "insertLength",
      "referenceDirectionHolderAngle",
      "directionOfSpindleRotation",
      "numberOfTeeth",
      "maxToolLife",
      "restToolLife",
      "toolLifeCount",
      "toolLifeAlarm"
    ],
    "description": "현재 스핀들에 장착되어 가공에 사용 중인 활성 공구의 위치, 이름, T-Code 정보를 제공합니다. 공구 날(Edge)별 형상/마모 보정값(길이, 반경), 툴 타입, 팁/홀더 각도 제원 및 잔여 수명 데이터를 상세히 조회할 수 있습니다."
  },
  {
    "category": "현재 실행 중인 NC 프로그램의 상태 정보",
    "keywords": [
      "sequenceNumber",
      "currentBlockCounter",
      "lastBlock",
      "currentBlock",
      "nextBlock",
      "activePartProgram",
      "programMode",
      "currentWorkOffsetIndex",
      "currentWorkOffsetCode",
      "currentDepthLevel",
      "modalIndex",
      "modalCode",
      "blockCounter",
      "programName",
      "depthLevel",
      "blockData",
      "searchType",
      "mainProgramName",
      "workOffsetIndex",
      "workOffsetValue",
      "workOffsetRotation",
      "workOffsetScalingFactor",
      "workOffsetMirroringEnabled",
      "programPath",
      "programSize",
      "programDate",
      "programNameWithPath",
      "singleBlock",
      "dryRun",
      "optionalStop",
      "blockSkip",
      "machineLock"
    ],
    "description": "현재 실행 중인 메인/서브 프로그램의 파일 경로, 시퀀스 번호(N 코드), 현재/전/후 블록의 G-코드 내용을 실시간으로 추적합니다. 활성화된 모달(Modal) 정보, 중단점, 그리고 싱글 블록/드라이 런 같은 실행 제어 옵션 상태를 포함합니다."
  },
  {
    "category": "공작물 좌표계의 오프셋 정보",
    "keywords": [
      "workOffsetValue",
      "workOffsetRotation",
      "workOffsetScalingFactor",
      "workOffsetMirroringEnabled",
      "workOffsetFine"
    ],
    "description": "G54, G55 등 설정된 공작물 좌표계의 각 축별 오프셋 값, 좌표 회전량(Rotation), 스케일링 비율, 미러링 적용 여부 및 Fine 오프셋 등 가공 원점 보정에 필요한 정밀 데이터를 조회합니다."
  },
  {
    "category": "알람 정보",
    "keywords": [
      "alarm",
      "alarmText",
      "alarmCategory",
      "alarmNumber",
      "raisedTimeStamp"
    ],
    "description": "장비 운영 중 발생한 모든 알람에 대해 알람 번호, 분류 카테고리(Category), 상세 메시지 텍스트(Text), 그리고 알람이 발생한 정확한 시점(Timestamp) 정보를 리스트 형태로 제공하여 장애 원인 분석을 돕습니다."
  },
  {
    "category": "사용자 변수",
    "keywords": [
      "userVariable"
    ],
    "description": "NC 프로그램 내에서 공정 제어 로직이나 상태 저장을 위해 활용되는 사용자 정의 변수의 현재 값을 실시간으로 조회합니다."
  },
  {
    "category": "CNC 내부 PLC 메모리 데이터",
    "keywords": [
      "rbitBlock",
      "bitBlock",
      "rbyteBlock",
      "byteBlock",
      "rwordBlock",
      "wordBlock",
      "rdwordBlock",
      "dwordBlock",
      "rqwordBlock",
      "qwordBlock"
    ],
    "description": "CNC 내부 PLC 메모리의 특정 주소에 직접 접근하여 Bit, Byte, Word, DWord, QWord 등 다양한 데이터 타입의 값을 읽거나 모니터링할 수 있는 로우 레벨 데이터 접근 기능을 제공합니다."
  },
  {
    "category": "장비 공구 영역 정보",
    "keywords": [
      "toolAreaEnabled",
      "numberOfMagazines",
      "numberOfRegisteredTools",
      "numberOfLoadedTools",
      "numberOfToolGroups",
      "numberOfToolOffsets",
      "magazineEnabled",
      "magazineName",
      "numberOfRealLocations",
      "magazinePhysicalNumber",
      "locationNumber",
      "toolName",
      "toolNumber",
      "numberOfEdges",
      "toolEnabled",
      "magazineNumber",
      "sisterToolNumber",
      "toolLifeUnit",
      "toolGroupNumber",
      "toolUseOrderNumber",
      "toolStatus",
      "toolType",
      "lengthOffsetNumber",
      "geoLengthOffset",
      "wearLengthOffset",
      "radiusOffsetNumber",
      "geoRadiusOffset",
      "wearRadiusOffset",
      "edgeEnabled",
      "geoLengthOffsetZ",
      "wearLengthOffsetZ",
      "geoLengthOffsetY",
      "wearLengthOffsetY",
      "geoOffsetNumber",
      "wearOffsetNumber",
      "cuttingEdgePosition",
      "tipAngle",
      "holderAngle",
      "insertAngle",
      "insertWidth",
      "insertLength",
      "referenceDirectionHolderAngle",
      "directionOfSpindleRotation",
      "numberOfTeeth",
      "maxToolLife",
      "restToolLife",
      "toolLifeCount",
      "toolLifeAlarm"
    ],
    "description":"장비의 전체 매거진 구성 및 상태(활성화 여부, 포트 수)를 조회하고, 매거진에 탑재되거나 등록된 모든 공구의 목록과 각 공구별 상세 제원, 보정값, 수명 상태를 통합적으로 관리합니다."
  },
  {
    "category": "내장 센서 데이터의 시계열 수집 정보",
    "keywords": [
      "bufferEnabled",
      "numberOfStream",
      "statusOfStream",
      "modOfStream",
      "machineChannelOfStream",
      "periodOfStream",
      "triggerOfStream",
      "frequencyOfStream",
      "streamEnabled",
      "streamFrequency",
      "streamCategory",
      "streamSubcategory",
      "streamType",
      "streamStartBit",
      "streamEndBit",
      "value"
    ],
    "description": "고속 데이터 분석을 위해 설정된 버퍼(Buffer)와 스트림(Stream)의 상태 및 설정을 관리합니다. 밀리초(ms) 단위로 수집된 축/스핀들 부하, 전류 등의 시계열 센서 데이터를 조회하여 정밀 가공 분석을 지원합니다."
  }
]
"고속 데이터 분석을 위해 설정된 버퍼(Buffer)와 스트림(Stream)의 상태 및 설정을 관리합니다. 밀리초(ms) 단위로 수집된 축/스핀들 부하, 전류 등의 시계열 센서 데이터를 조회하여 정밀 가공 분석을 지원합니다."
  
"""
