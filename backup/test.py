from langgraph.graph import StateGraph
from dotenv import load_dotenv

print("LangGraph successfully installed!")
from langchain_anthropic import ChatAnthropic

# .env 파일에서 환경 변수
load_dotenv()

# Anthropic Claude 설정
claude = ChatAnthropic(
   model="claude-sonnet-4-20250514",
)

# Anthropic Claude 사용
claude_response = claude.invoke("Explain the concept of Direct Graph.")
print("Claude's response:", claude_response)
