from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# graph_logic.py에서 완성된 LangGraph chain을 가져옵니다.
# uvicorn을 프로젝트 루트에서 실행하므로 'backend.' 경로를 사용합니다.
from backend.graph_logic import chain as graph_chain

app = FastAPI(
    title="LangGraph Service",
    description="An API to interact with the LangGraph RAG agent.",
)

# Pydantic model for the request body
class InvokeRequest(BaseModel):
    question: str

# Pydantic model for the response body
class InvokeResponse(BaseModel):
    answer: str



@app.post("/invoke", response_model=InvokeResponse)
async def invoke_chain(request: InvokeRequest):
    """
    Endpoint to invoke the LangGraph chain.
    """
    question = request.question
    
    try:
        # LangGraph chain을 비동기적으로 호출합니다.
        # ainvoke는 최종 상태를 반환합니다.
        final_state = await graph_chain.ainvoke({"question": question})
        
        # 최종 답변이 있는지 확인하고 반환합니다.
        answer = final_state.get("final_answer")
        if answer is None:
            raise HTTPException(status_code=500, detail="Could not get a final answer from the agent.")
            
        return {"answer": answer}

    except Exception as e:
        # 에러가 발생하면 클라이언트에게 알립니다.
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# API 라우트 정의 후에 정적 파일을 마운트합니다.
# 이 코드는 서버의 루트 URL('/')로 들어오는 GET 요청을 'frontend' 폴더로 전달하고,
# 해당 경로에 맞는 파일이 없으면 index.html을 보여줍니다.
app.mount("/", StaticFiles(directory="frontend", html=True), name="static")

# To run this server, navigate to the `dt-agent` directory in your terminal and run:
# uvicorn backend.main:app --reload --port 8001
