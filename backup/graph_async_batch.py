import os
import json
import asyncio
import httpx
import logging
from typing import Annotated, List, Dict, Any, TypedDict, Union
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END

# 환경 변수 로드
load_dotenv()
# 기본 로깅은 끄고 커스텀 출력만 봅니다
logging.basicConfig(level=logging.CRITICAL)

# -------------------------------------------------------------------------
# 1. 디버깅 및 로깅 유틸리티 (핵심 추가 사항)
# -------------------------------------------------------------------------

def load_json_file(file_path: Path) -> Dict:
    """JSON 파일을 로드하는 유틸리티 함수"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.critical(f"치명적 오류: 필수 설정 파일({file_path})을 찾을 수 없습니다.")
        raise # 예외를 다시 발생시켜 프로그램 중단
    except json.JSONDecodeError:
        logging.critical(f"치명적 오류: 설정 파일({file_path})의 JSON 형식이 잘못되었습니다.")
        raise # 예외를 다시 발생시켜 프로그램 중단

class DebugCallbackHandler(BaseCallbackHandler):
    """LLM 실행 시 토큰 사용량과 내부 동작을 캡처하여 출력하는 핸들러"""
    
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> None:
        print("\n" + "🔵 " * 10 + " [LLM INPUT PROMPT] " + "🔵 " * 10)
        for msg in messages[0]:
            content = str(msg.content)
            # 매뉴얼이 너무 길면 축약해서 보여줌
            if len(content) > 1500:
                display_content = content[:500] + "\n... (중략: 매뉴얼 본문) ...\n" + content[-500:]
            else:
                display_content = content
            print(f"[{msg.type.upper()}]: {display_content}")
        print("🔵 " * 25 + "\n")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
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
    """현재 State의 상태를 깔끔하게 출력"""
    print(f"\n" + "⚡ " * 15)
    print(f"⚡ [STATE DUMP] Node: {node_name}")
    print(f"⚡ " * 15)

    keys_to_show = ["current_stage_index", "api_results", "final_answer"]

    if "endpoint_plan" in state and state["endpoint_plan"]:
        plan = state["endpoint_plan"]
        if plan:
            print(f"  - endpoint_plan: {len(plan)} stages loaded")
    
    if "executable_stage" in state and state["executable_stage"]:
        steps = state["executable_stage"].get('steps', [])
        print(f"  - executable_stage: {len(steps)} steps ready for execution")

    for k in keys_to_show:
        if k in state and state[k] is not None:
            val = state[k]
            if k == "api_results" and val:
                print(f"  - api_results: {list(val.keys())}")
            elif val is not None:
                print(f"  - {k}: {val}")
    print("-" * 60 + "\n")


# -------------------------------------------------------------------------
# 2. 설정 및 데이터 모델
# -------------------------------------------------------------------------

MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manual', 'torus.md')
TORUS_MD_CONTENT = ""
try:
    with open(MD_PATH, 'r', encoding='utf-8') as f:
        TORUS_MD_CONTENT = f.read()
except FileNotFoundError:
    print("⚠️ 매뉴얼 파일을 찾을 수 없습니다.")

JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'manual', 'uri_params.json')
PARAMS_JSON = load_json_file(JSON_PATH)

# --- Endpoint-Planner의 출력 모델 ---
class ApiStep(BaseModel):
    step_id: str = Field(description="단계 식별자 (예: '1-1', '1-2')")
    endpoint: str = Field(description="호출할 API 엔드포인트")
    reasoning: str = Field(description="이 단계의 실행 목적")
    required_params_desc: str = Field(description="필요한 파라미터 힌트")

class ParallelStage(BaseModel):
    stage_id: int = Field(description="실행 순서 (1부터 시작)")
    steps: List[ApiStep] = Field(description="이 단계에서 병렬로 실행할 엔드포인트 리스트")

class EndpointExecutionPlan(BaseModel):
    plan: List[ParallelStage] = Field(description="순차적으로 실행될 스테이지 리스트")

# --- Param-Planner의 출력 모델 ---
class ExecutableStep(BaseModel):
    step_id: str = Field(description="고유 식별자 (예: '3-1-instance-1')")
    endpoint: str = Field(description="호출할 API 엔드포인트")
    params: Dict[str, Any] = Field(description="API 호출에 사용할 실제 파라미터")
    reasoning: str = Field(description="이 특정 API 호출을 실행하는 이유")

class ExecutableStage(BaseModel):
    steps: List[ExecutableStep] = Field(description="이번 스테이지에서 병렬로 실행할 API 호출 목록")

# --- 메인 Agent State ---
class AgentState(TypedDict):
    question: str
    endpoint_plan: List[Dict] 
    executable_stage: Dict 
    current_stage_index: int
    api_results: Dict
    final_answer: str

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
                data = response.json()
                print(f"   📥 [API Response] Success from {url} | Params: {params}")
                return data
            else:
                print(f"   ⚠️ [API Response] Failed from {url} | Params: {params} | Status: {response.status_code}")
                return {"error": f"Status {response.status_code}", "body": response.text}
    except Exception as e:
        print(f"   🔥 [API Response] Exception from {url} | Params: {params} | Error: {e}")
        return {"error": str(e)}

class TorusDynamicAgent:
    def __init__(self, model):
        self.debug_handler = DebugCallbackHandler()
        self.model = model
        self.endpoint_planner_model = model.with_structured_output(EndpointExecutionPlan)
        self.param_planner_model = model.with_structured_output(ExecutableStage)
        self.PARAMS_JSON = PARAMS_JSON

    async def get_params_info(self, endpoint_list: List[str]):
        results = {}
        for endpoint in endpoint_list:
            endpoint_info = self.PARAMS_JSON.get(endpoint)
            results[endpoint] = endpoint_info.get("required_params") if endpoint_info else "매뉴얼 참조 필요"
        return results

    async def endpoint_planner_node(self, state: AgentState):
        print_state_debug("Endpoint Planner", state)
        question = state["question"]
        system_text = (
            "당신은 TORUS API 병렬 처리 설계자입니다. 매뉴얼을 참고하여 실행 계획을 수립하되, "
            "**서로 의존성이 없는 API 호출은 같은 스테이지(Stage)로 묶어야 합니다.**\n\n"
            "=== 작성 규칙 ===\n"
            "1. **Parallel Stage**: 이전 단계의 결과가 필요하지 않고, 서로 독립적인 API들은 하나의 Stage 안에 리스트로 넣으세요.\n"
            "2. **Sequential Stage**: 이전 단계의 데이터가 반드시 필요한 경우에만 다음 스테이지로 분리하세요.\n\n"
            f"[API 매뉴얼]\n{TORUS_MD_CONTENT}"
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
        if current_idx >= len(endpoint_plan):
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
            "1. **반복 실행**: 만약 이전 데이터에 리스트(예: 장비 20개)가 있고, 다음 계획이 그 리스트의 각 항목에 대해 실행되어야 한다면, 리스트의 모든 항목에 대한 API 호출을 **개별적으로 모두 생성**해야 합니다.\n"
            "2. **파라미터 채우기**: '필수 파라미터 정보'와 '이전 데이터'를 보고 각 API 호출에 필요한 파라미터 값을 정확히 채워넣으세요.\n"
            "3. **ID 생성**: 각 `step_id`는 `{원래 step_id}-instance-{n}` 형식으로 고유하게 만드세요. (예: '3-1-instance-1')\n"
            "4. **JSON 출력**: 반드시 `ExecutableStage` 모델의 JSON 형식으로만 출력하세요.\n\n"
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
        return {"executable_stage": executable_stage_dict, "current_stage_index": current_idx + 1}

    async def _execute_single_step(self, step_info: Dict[str, Any]):
        step_id = step_info['step_id']
        endpoint = step_info['endpoint']
        params = step_info.get('params', {})
        result_data = await call_torus_api(endpoint, params)
        return {f"step_{step_id}": {"request_endpoint": endpoint, "request_params": params, "response_data": result_data}}

    async def executor_node(self, state: AgentState):
        print_state_debug("Executor", state)
        executable_stage = state.get("executable_stage", {})
        steps_to_run = executable_stage.get('steps', [])
        if not steps_to_run:
            print("   -> No steps to execute.")
            return {}
        
        api_results = state.get("api_results", {})
        print(f"\n🚀 [Executing Stage] - {len(steps_to_run)} parallel API calls")
        tasks = [self._execute_single_step(step_info) for step_info in steps_to_run]
        stage_results_list = await asyncio.gather(*tasks)
        
        for res_dict in stage_results_list:
            api_results.update(res_dict)
        return {"api_results": api_results}

    async def synthesizer_node(self, state: AgentState):
        print_state_debug("Synthesizer", state)
        question = state["question"]
        results = state["api_results"]
        final_prompt = (
            "수집된 API 데이터를 바탕으로 사용자에게 최종 답변을 한국어로 작성하세요.\n\n"
            f"질문: {question}\n"
            f"데이터: {json.dumps(results, ensure_ascii=False, indent=2)}"
        )
        response = await self.model.ainvoke(
            [HumanMessage(content=final_prompt)],
            config={"callbacks": [self.debug_handler]}
        )
        return {"final_answer": response.content}

    def should_continue(self, state: AgentState):
        executable_stage = state.get("executable_stage", {})
        if not executable_stage or not executable_stage.get("steps"):
            print("--> All stages complete. Proceeding to Synthesizer.")
            return "synthesizer"
        else:
            print("--> Steps generated. Proceeding to Executor.")
            return "executor"

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("endpoint_planner", self.endpoint_planner_node)
        workflow.add_node("param_planner", self.param_planner_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        
        workflow.set_entry_point("endpoint_planner")
        workflow.add_edge("endpoint_planner", "param_planner")
        workflow.add_conditional_edges(
            "param_planner",
            self.should_continue,
            {"executor": "executor", "synthesizer": "synthesizer"}
        )
        workflow.add_edge("executor", "param_planner")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()

# -------------------------------------------------------------------------
# 4. Main 실행
# -------------------------------------------------------------------------
async def main():
    llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-20250514")

    bot = TorusDynamicAgent(model=llm)
    graph = bot.build_graph()
    
    query = "지멘스에 등록된 1번 공구의 수명 정보를 알려줘"
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
