# mcp_client.py (또는 agent_setup.py 등)
from langchain_core.language_models.llms import LLM
from typing import Any, List, Mapping, Optional, Dict
import requests # LLM API 서버 호출용
import os
import asyncio

# ... (기존 mcp, langchain_mcp_adapters, langgraph, dotenv 임포트)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# --- 로컬 LLM API 호출을 위한 커스텀 Langchain LLM 래퍼 ---
class CustomLocalLLM(LLM):
    # FastAPI 서버 주소
    api_url: str = "http://localhost:8000/query_rag" # llm_server.py가 실행되는 주소

    @property
    def _llm_type(self) -> str:
        return "custom_local_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None, # Langchain 표준 인터페이스 (여기서는 간단히 무시)
        **kwargs: Any, # temperature, top_p 등을 받을 수 있음
    ) -> str:
        payload = {
            "prompt": prompt,
            "max_new_tokens": kwargs.get("max_new_tokens", 256),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9)
        }
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status() # 오류 발생 시 예외 발생
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            print(f"LLM API 호출 오류: {e}")
            return f"LLM API 호출 중 오류 발생: {e}" # 실제로는 더 나은 오류 처리 필요
        except KeyError:
            print(f"LLM API 응답 형식 오류: {response.text}")
            return "LLM API 응답 형식 오류"


    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        # 비동기 HTTP 클라이언트 (예: aiohttp)를 사용하면 더 좋지만,
        # 간단하게 동기 호출을 비동기 컨텍스트에서 실행합니다.
        # (실제 프로덕션에서는 aiohttp 등을 고려하세요)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call, prompt, stop, **kwargs)


    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"api_url": self.api_url}

# --- 기존 main 함수 수정 ---
async def main():
    # model = ChatGoogleGenerativeAI(...) # 기존 Gemini 모델 대신
    model = CustomLocalLLM(api_url="http://localhost:8000/generate") # 로컬 LLM API 사용

    server_params = StdioServerParameters(
        command="python", # 또는 mcp_server.py를 직접 실행 가능하게 만들었다면 해당 명령
        args=["mcp_server.py"], # 경로 수정 필수
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session) # MCP 서버의 도구 로드
            agent = create_react_agent(model, tools) # 로컬 LLM과 MCP 도구를 사용하는 에이전트

            # RAG를 위한 Pinecone 설정 (만약 사용한다면)
            # pinecone_api_key = os.environ.get("PINECONE_API_KEY")
            # pinecone_environment = os.environ.get("PINECONE_ENVIRONMENT")
            # index_name = "your-pinecone-index-name"
            # if pinecone_api_key and pinecone_environment and index_name:
            # pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
            # vector_store = Pinecone.from_existing_index(index_name, OpenAIEmbeddings()) # 또는 다른 임베딩 모델
            # retriever = vector_store.as_retriever()
            # 이제 이 retriever를 agent나 chain에 통합할 수 있습니다.
            # 예를 들어, agent가 특정 질문에 대해 Pinecone에서 정보를 검색하도록 도구를 추가하거나,
            # 질문 처리 파이프라인의 일부로 만들 수 있습니다.

            print("에이전트가 준비되었습니다. 질문을 입력하세요 (종료: exit).")
            while True:
                user_input = await asyncio.to_thread(input, "나: ")
                if user_input.lower() == "exit":
                    break
                if not user_input.strip():
                    continue

                # result = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"}) # 이전 예시
                # Langgraph create_react_agent는 "messages" 키에 대화 히스토리를 리스트로 받습니다.
                # 사용자의 새 메시지를 추가하여 전달해야 합니다.
                # 간단한 단일 턴 상호작용의 경우:
                messages = [("user", user_input)] # Langgraph ReactAgent의 입력 형식
                result = await agent.ainvoke({"messages": messages})

                # result 딕셔너리에는 'messages' 키로 전체 대화 히스토리(AI 답변 포함)가 들어있습니다.
                # 마지막 AI 답변만 추출하려면:
                ai_response = result['messages'][-1].content if result['messages'] and result['messages'][-1].type == 'ai' else "No AI response found."
                print(f"에이전트: {ai_response}")

if __name__ == "__main__":
    load_dotenv()
    asyncio.run(main())