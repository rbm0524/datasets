from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI

#Langchain Agent

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")

model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # 사용하고자 하는 Gemini 모델명으로 변경
                                    # 예: "gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro-latest" 등
    google_api_key=api_key
)

server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],  # 경로 수정 필수
)

# 비동기 작업을 수행할 main 함수 정의
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)
            result = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            print(result) # 결과를 확인하기 위해 추가

# 스크립트가 직접 실행될 때 main 함수를 실행
if __name__ == "__main__":
    asyncio.run(main())