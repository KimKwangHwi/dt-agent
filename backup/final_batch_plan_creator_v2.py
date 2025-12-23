import os
import json
import asyncio
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic

# --- 기본 설정 ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# --- 매뉴얼 및 파라미터 정보 로드 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(current_dir, 'manual', 'torus.md')
JSON_PATH = os.path.join(current_dir, 'manual', 'uri_params.json')

try:
    with open(MD_PATH, 'r', encoding='utf-8') as f:
        TORUS_MD_CONTENT = f.read()
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        PARAMS_JSON = json.load(f)
except FileNotFoundError as e:
    logging.critical(f"⚠️ 필수 파일({e.filename})을 찾을 수 없습니다. 스크립트를 종료합니다.")
    exit()

# --- Pydantic 모델 정의 ---
class ApiStep(BaseModel):
    step_id: str
    endpoint: str
    reasoning: str
    # lookup_params는 후처리로 추가되므로 모델에서 제외

class ParallelStage(BaseModel):
    stage_id: int
    steps: List[ApiStep]

class EndpointExecutionPlan(BaseModel):
    plan: List[ParallelStage]

class QuestionAndPlan(BaseModel):
    question: str
    execution_plan: EndpointExecutionPlan

class BatchEndpointExecutionPlans(BaseModel):
    batch_plans: List[QuestionAndPlan]

# --- 메인 로직 클래스 ---
class EnhancedBatchPlanGenerator:
    def __init__(self, model):
        self.batch_planner_model = model.with_structured_output(BatchEndpointExecutionPlans)

    async def _get_raw_plans_from_llm(self, questions: List[str]) -> List[Dict]:
        """LLM을 호출하여 구조화된 계획 초안을 받습니다."""
        system_prompt = (
            "당신은 여러 개의 사용자 질문을 한 번에 처리하여 각각에 대한 TORUS API 실행 계획을 수립하는 고효율 AI 설계자입니다.\n"
            "주어진 API 매뉴얼을 참고하여, 모든 질문에 대한 계획을 각각 생성해야 합니다.\n\n"
            "## 핵심 규칙: API 의존성 분석\n"
            "1.  **파라미터 의존성**: 많은 API 엔드포인트는 `machine`, `channel`, `axis` 같은 ID를 파라미터로 요구합니다.\n"
            "2.  **ID 획득**: 이런 ID들은 다른 API를 호출해야만 얻을 수 있습니다. 예를 들어, `machine` ID는 `/machine/list`를 통해 얻고, 특정 machine의 `channel` 수는 `/machine/numberOfChannels`를 통해 얻습니다.\n"
            "3.  **단계적 계획 수립**: 따라서, 어떤 API를 호출하기 위해 필요한 파라미터 값을 다른 API를 통해 얻어야 한다면, **반드시 선행 API 호출을 이전 스테이지(Stage)에 배치해야 합니다.** 이는 모든 계획 수립에서 가장 중요한 규칙입니다.\n"
            "4.  **예시**: '1번 장비의 2번 채널에 연결된 세 번째 축의 부하'를 조회하려면, 최종적으로 `/machine/channel/axis/axisLoad` API가 필요하며 `machine=1`, `channel=2`, `axis=3` 파라미터가 필요합니다. 이 경우 계획은 아래와 같이 의존성을 고려하여 수립되어야 합니다.\n"
            "    - **Stage 1**: `/machine/list` (질문에서 '1번 장비'를 특정했더라도, 시스템은 정확한 ID를 알아야 하므로 이 단계가 필요합니다.)\n"
            "    - **Stage 2**: `/machine/numberOfChannels` (`machine=1` 파라미터를 사용하여 채널 수를 확인합니다.)\n"
            "    - **Stage 3**: `/machine/channel/numberOfAxes` (`machine=1`, `channel=2` 파라미터를 사용하여 축 개수를 확인합니다.)\n"
            "    - **Stage 4**: `/machine/channel/axis/axisLoad` (`machine=1`, `channel=2`, `axis=3` 파라미터를 사용하여 최종 정보를 조회합니다.)\n\n"
            "## 기타 규칙\n"
            "-   **병렬화**: 한 계획 내에서 서로 의존성이 없는 API 호출은 같은 스테이지(Stage)로 묶어 효율성을 높이세요.\n"
            "-   **구조 준수**: 최종 응답은 반드시 `BatchEndpointExecutionPlans` JSON 스키마에 맞춰야 합니다.\n\n"
            f"--- API 매뉴얼 ---\n{TORUS_MD_CONTENT}\n---"
        )
        formatted_questions = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
        human_prompt = f"다음은 처리해야 할 모든 질문 목록입니다. 위에 명시된 '핵심 규칙: API 의존성 분석'을 반드시 준수하여 각 질문에 대한 실행 계획을 생성해주세요:\n\n{formatted_questions}"
        
        logging.info(f"LLM 호출: {len(questions)}개 질문에 대한 일괄 계획 생성 시작...")
        
        try:
            batch_result = await self.batch_planner_model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            )
            logging.info("LLM 호출 완료. 계획 후처리 시작...")
            return [plan.model_dump() for plan in batch_result.batch_plans]
        except Exception as e:
            logging.error(f"일괄 계획 생성 중 심각한 오류 발생: {e}")
            return []

    def _get_params_info(self, endpoint: str) -> str:
        """단일 엔드포인트에 대한 파라미터 정보를 조회합니다."""
        endpoint_info = PARAMS_JSON.get(endpoint)
        return endpoint_info.get("required_params") if endpoint_info else "존재하지 않는 엔드포인트"

    def _post_process_plans(self, raw_plans: List[Dict]) -> List[Dict]:
        """
        LLM이 생성한 계획을 후처리합니다.
        1. step_id를 '{stage_id}-{step_index}' 형식으로 표준화합니다.
        2. 각 단계에 lookup_params 정보를 추가합니다.
        """
        processed_plans = []
        for plan_data in raw_plans:
            plan = plan_data['execution_plan']['plan']
            
            # step_id 포맷팅 및 lookup_params 추가 로직
            for stage in plan:
                stage_id = stage['stage_id']
                for i, step in enumerate(stage['steps'], 1):
                    # 1. step_id 표준화
                    step['step_id'] = f"{stage_id}-{i}"
                    
                    # 2. lookup_params 추가
                    step['lookup_params'] = self._get_params_info(step['endpoint'])

            processed_plans.append(plan_data)
        
        logging.info("모든 계획에 대한 후처리(step_id 표준화, lookup_params 추가) 완료.")
        return processed_plans

    async def generate_and_process_plans(self, questions: List[str]) -> List[Dict]:
        """전체 프로세스를 실행: LLM 호출 -> 후처리"""
        raw_plans = await self._get_raw_plans_from_llm(questions)
        if not raw_plans:
            return []
        
        processed_plans = self._post_process_plans(raw_plans)
        return processed_plans


async def main():
    output_file_path = os.path.join(current_dir, 'generated_faiss_db_final.json')
    questions_file_path = os.path.join(current_dir, 'manual', 'questions.txt')

    try:
        with open(questions_file_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        logging.info(f"'{questions_file_path}'에서 {len(questions)}개의 질문을 로드했습니다.")
    except FileNotFoundError:
        logging.critical(f"질문 파일 '{questions_file_path}'을 찾을 수 없습니다.")
        return

    if not questions:
        logging.warning("질문 파일에 내용이 없습니다.")
        return

    llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-20250514")
    planner = EnhancedBatchPlanGenerator(model=llm)
    
    # LLM 호출 및 후처리가 포함된 최종 계획 리스트를 받음
    final_plans_list = await planner.generate_and_process_plans(questions)
    
    # faiss_db.json 형식으로 변환
    if final_plans_list:
        faiss_db_format_dict = {
            item["question"]: item["execution_plan"]["plan"]
            for item in final_plans_list
        }
        
        # 변환된 딕셔너리를 JSON 파일로 저장
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(faiss_db_format_dict, f, ensure_ascii=False, indent=4)
        
        logging.info(f"\n{'='*60}\n✅ 최종 보정된 계획을 '{output_file_path}' 파일에 저장했습니다.\n{'='*60}")
    else:
        logging.error("생성된 계획이 없거나 오류가 발생하여 파일을 저장하지 않았습니다.")

if __name__ == "__main__":
    asyncio.run(main())
