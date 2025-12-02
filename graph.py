import operator
from dotenv import load_dotenv
from typing import Annotated, List, Dict, Optional, Any, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from pathlib import Path
import json
import logging
import os


md_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'manual',
    'torus.md'
)
with open(md_path, 'r', encoding='utf-8') as f :
    TORUS_MD_CONTENT = f.read()


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




# --- 1. 상태 정의 (State) ---
class AgentState(TypedDict):
    question: str                   # 사용자 질문
    detected_endpoints: List[str]   # LLM이 추출한 엔드포인트 리스트
    final_result: Dict[str, Any]    # 최종 결과 (엔드포인트: 파라미터 정보)

# --- 2. LLM 구조화 출력을 위한 스키마 정의 ---
class EndpointList(BaseModel):
    """List of API endpoints extracted from the documentation based on user query."""
    endpoints: List[str] = Field(
        description="A list of API endpoints (e.g., '/machine/list', '/machine/channel/activeTool/toolName')."
    )

# --- 3. 에이전트 클래스 정의 ---
class TorusAgent:
    
    PARAMS_JSON = load_json_file(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'manual',
            'uri_params.json'
        )
    )
    
    def __init__(self, model):
        self.model = model
        # 사용자의 get_params_info 로직에서 참조할 JSON 데이터
        
    # --- 사용자 제공 함수 (get_params_info) ---
    async def get_params_info(self, endpoint_list: List[str]):
        """
        사용자 질문에 들어오고 장비 목록이 반환된 후, 엔드포인트에 대한 필수 파라미터 정보를 조회합니다.
        """
        results = {}
        
        for endpoint in endpoint_list:
            # 실제로는 self.PARAMS_JSON에서 조회
            endpoint_info = self.PARAMS_JSON.get(endpoint)

            # 값이 존재할 경우에만 required_params를 찾습니다.
            if endpoint_info:
                params_info = endpoint_info.get("required_params")
            else:
                params_info = None  # 키가 없는 경우 None으로 처리
                
            results[endpoint] = params_info
                
        return results

    # --- Node 1: 엔드포인트 추출 (LLM) ---
    async def identify_endpoints_node(self, state: AgentState):
        question = state["question"]
        docs = TORUS_MD_CONTENT

        # LLM에게 구조화된 출력을 강제 (endpoints 리스트만 뽑도록)
        structured_llm = self.model.with_structured_output(EndpointList)

        system_prompt = (
            "당신은 TORUS API에 대한 전문가입니다."
            "제공된 문서를 기반으로, 필요한 API 엔드포인드들을 모두 식별하세요."
            "사용자의 질문에 답변하세요. "
            "Return ONLY the list of exact endpoint paths found in the document."
        )

        user_prompt = f"""
        [Documentation]
        {docs}

        [User Question]
        {question}

        [Instruction]
        사용자 질문을 처리하기 위해 필요한 메뉴얼 내에 존재하는 모든 엔드포인트를 생성하시오.
        """

        # LLM 호출
        response = await structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        return {"detected_endpoints": response.endpoints}

    # --- Node 2: 파라미터 정보 조회 (Tool/Logic) ---
    async def fetch_params_node(self, state: AgentState):
        endpoints = state["detected_endpoints"]
        
        # 사용자가 제공한 함수 호출
        params_info = await self.get_params_info(endpoints)
        
        return {"final_result": params_info}

    # --- 그래프 빌드 ---
    def build_graph(self):
        workflow = StateGraph(AgentState)

        # 노드 추가
        workflow.add_node("identify_endpoints", self.identify_endpoints_node)
        workflow.add_node("fetch_parameters", self.fetch_params_node)

        # 엣지 연결 (Linear Flow)
        workflow.set_entry_point("identify_endpoints")
        workflow.add_edge("identify_endpoints", "fetch_parameters")
        workflow.add_edge("fetch_parameters", END)

        return workflow.compile()

# --- 4. 실행 예시 ---

# (1) Mock Data: 실제로는 파일에서 로드하거나 DB에서 가져올 PARAMS_JSON
# 예시 동작을 위해 torus.md에 기반한 일부 매핑 데이터를 생성합니다.
# (2) torus.md 내용 (Context) - 프롬프트에 제공된 내용



async def main():
    # LLM 설정 (OpenAI 키 설정 필요)
    llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-20250514")

    # 에이전트 초기화
    agent = TorusAgent(model=llm)
    graph = agent.build_graph()

    # 사용자 질문
    user_query = "등록된 장비의 현재 활성화된 공구의 공구 이름을 모두 알려줘"

    # 그래프 실행
    print(f"User Query: {user_query}")
    print("-" * 50)
    
    async for event in graph.astream({
        "question": user_query,
        "torus_docs": TORUS_MD_CONTENT
    }):
        for key, value in event.items():
            print(f"\n[Node Finished: {key}]")
            if key == "identify_endpoints":
                print(f" -> Extracted Endpoints: {value['detected_endpoints']}")
            elif key == "fetch_parameters":
                print(f" -> Final Params Mapping: {value['final_result']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
