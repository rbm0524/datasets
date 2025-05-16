from typing import List
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    # 아래 내용을 docstring이라고 함. llm이 이해하기 쉽게 꼭 작성하기
    """
    두 정수 a와 b를 더한 결과를 반환합니다.
    예: add(3, 5) -> 8
    """
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio") # sse 방식도 있음
# @mcp.prompt() : llm과의 상호작용을 위한 템플릿. message를 받아 llm에 전달할 지시문 제공