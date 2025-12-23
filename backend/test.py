import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

print("--- Claude API 캐싱 기능 테스트 (수정 버전) ---")

# 충분히 긴 시스템 메시지 생성 (최소 4096 토큰 이상)
long_system_content = """당신은 초등학교 교사입니다. 
학생들에게 복잡한 과학 개념을 쉽고 재미있게 설명하는 것이 당신의 임무입니다.
항상 친절하고 인내심 있게 답변하며, 학생들의 눈높이에 맞춰 설명합니다.
비유와 예시를 많이 사용하고, 일상생활과 연결지어 설명합니다.
어려운 용어는 피하고, 간단한 단어로 풀어서 설명합니다.
질문을 받으면 먼저 칭찬하고, 격려하면서 답변합니다.
""" * 30  # 충분히 긴 컨텍스트 생성 (4096 토큰 이상)

# 방법: 딕셔너리 형태로 메시지 구성 (이게 핵심!)
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": long_system_content,
                "cache_control": {"type": "ephemeral"}  # 여기에 cache_control 추가
            }
        ],
    },
    {
        "role": "user",
        "content": "상대성 이론에 대해 간단히 설명해줘.",
    },
]

# LangChain ChatAnthropic 초기화
llm = ChatAnthropic(
    model="claude-haiku-4-5",
    temperature=0
)

print("\n첫 번째 호출 (캐시 생성)...")
response_1 = llm.invoke(messages)
usage_1 = response_1.usage_metadata

print("\n두 번째 호출 (캐시 사용)...")
response_2 = llm.invoke(messages)
usage_2 = response_2.usage_metadata

print(f"\n=== 결과 비교 ===")
print(f"\n첫 번째 호출:")
print(f"  Input tokens: {usage_1.get('input_tokens', 0)}")
print(f"  Cache creation: {usage_1.get('input_token_details', {}).get('cache_creation', 0)}")
print(f"  Cache read: {usage_1.get('input_token_details', {}).get('cache_read', 0)}")

print(f"\n두 번째 호출:")
print(f"  Input tokens: {usage_2.get('input_tokens', 0)}")
print(f"  Cache creation: {usage_2.get('input_token_details', {}).get('cache_creation', 0)}")
print(f"  Cache read: {usage_2.get('input_token_details', {}).get('cache_read', 0)}")

# 캐싱이 작동했는지 확인
cache_read = usage_2.get('input_token_details', {}).get('cache_read', 0)
if cache_read > 0:
    print(f"\n✅ 캐싱이 정상적으로 작동했습니다!")
    print(f"   절약된 토큰: {cache_read} 토큰")
else:
    print("\n❌ 캐싱이 작동하지 않았습니다.")
    print("\n가능한 원인:")
    print("1. 시스템 메시지가 4096 토큰 미만일 수 있습니다.")
    print("2. 5분 이내에 같은 요청을 해야 캐시가 유지됩니다.")
    print("3. 메시지 형식이 정확하지 않을 수 있습니다.")