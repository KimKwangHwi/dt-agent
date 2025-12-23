# 프로젝트 개요

이 프로젝트는 Python 기반의 백엔드와 웹 프론트엔드를 포함하는 풀스택 애플리케이션입니다. 벡터 데이터베이스(FAISS)를 활용하여 효율적인 데이터 검색 및 처리 기능을 제공하며, 그래프 로직을 통해 복잡한 데이터 관계를 관리하고 활용하는 데 중점을 둡니다.

## 주요 구성 요소

### 백엔드 (Python)
-   `main.py`: 애플리케이션의 메인 진입점입니다.
-   `graph_logic.py`: 그래프 기반의 데이터 처리 로직을 구현합니다.
-   `graph_async_batch_vectorDB.py`: 비동기 배치 처리와 벡터 데이터베이스 연동을 담당합니다.
-   `final_batch_plan_creator.py`: 최종 배치 계획을 생성하는 기능을 수행합니다. (테스트용. 사용하지 않음)
-   `manage_db.py`: 데이터베이스 관리와 관련된 유틸리티 함수들을 포함합니다.
-   `test.py`: 백엔드 기능에 대한 테스트 코드를 포함합니다.
-   `manual/`: 수동 테스트 데이터, 질문, URI 파라미터 등 추가 자료를 포함합니다.

### 프론트엔드 (웹)
-   `index.html`: 애플리케이션의 메인 웹 페이지입니다.
-   `script.js`: 클라이언트 측 상호작용 및 로직을 담당합니다.
-   `styles.css`: 애플리케이션의 스타일을 정의합니다.

### 데이터
-   `faiss_db.json`, `faiss_db_permanent.json`, `faiss_index.bin`: FAISS (Facebook AI Similarity Search) 라이브러리를 사용하여 생성된 벡터 데이터베이스 파일들입니다. 효율적인 유사성 검색을 위해 사용됩니다.

### 기타
-   `requirements.txt`: Python 종속성 목록입니다.
-   `myenv/`: 프로젝트를 위한 Python 가상 환경입니다.

## 프로젝트 실행 방법 (예시)

1.  **가상 환경 설정 및 종속성 설치:**
    ```bash
    python -m venv myenv
    ./myenv/Scripts/activate # Windows
    # source myenv/bin/activate # macOS/Linux
    pip install -r requirements.txt
    ```

2.  **백엔드 실행:**
    ```
    uvicorn backend.main:app --reload
    mock_server와 동시에 실행하는 경우 port 번호를 달리해야 합니다.
    ```

3.  **프론트엔드 접근:**
    백엔드가 실행된 후, 웹 브라우저에서 `frontend/index.html` 파일을 열거나, 적절한 웹 서버를 통해 접근합니다.

## 잠재적 활용 분야

이 프로젝트는 질문 응답 시스템, 지식 그래프 탐색, 추천 시스템, 정보 검색 등 벡터 데이터와 그래프 구조를 활용하는 다양한 분야에 적용될 수 있습니다. 특히, 대규모 데이터셋에서 효율적인 유사성 검색 및 복잡한 관계 추론이 필요한 경우에 유용합니다.