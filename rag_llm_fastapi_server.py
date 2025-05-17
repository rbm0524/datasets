from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
from pinecone import Pinecone
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# 기본 로거 가져오기
logger = logging.getLogger(__name__)

# 로깅 레벨 설정 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러 생성 및 포매터 설정
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# 로거에 핸들러 추가
logger.addHandler(console_handler)

load_dotenv()

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "MLP-KTLim/llama-3-Korean-Bllossom-8B")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./lora-llama-3-korean-results_2")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
device = "cuda" if torch.cuda.is_available() else "cpu"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

EMBEDDING_MODEL_NAME = "llama-text-embed-v2"


print(f"PINECONE_API_KEY: {PINECONE_API_KEY}")
print(f"PINECONE_ENVIRONMENT: {PINECONE_ENVIRONMENT}")
print(f"PINECONE_INDEX_NAME: {PINECONE_INDEX_NAME}")

app = FastAPI()

# 전역 변수
llm_model = None
tokenizer = None
pc = None
index = None
langchain_llm = None
llm_chain = None

# LLM 모델을 LangChain용으로 감쌈
def wrap_llm_with_langchain(model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9
    )
    return HuggingFacePipeline(pipeline=pipe)


# --- 모델 로드 함수 ---
def load_llm_and_tokenizer(base_model_id, lora_adapter_path, quant_config):
    print(f"기본 LLM 모델 로드 중: {base_model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if quant_config else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        print(f"토크나이저 로드 중: {base_model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if lora_adapter_path and os.path.exists(lora_adapter_path):
            print(f"LoRA 어댑터 로드 및 병합 중: {lora_adapter_path}...")
            model = PeftModel.from_pretrained(model, lora_adapter_path)
            model = model.merge_and_unload()
            print("LoRA 어댑터 병합 완료.")
        else:
            print("LoRA 어댑터 경로가 없거나 잘못되어 기본 모델만 사용합니다.")

        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"LLM 또는 토크나이저 로드 중 오류: {e}")
        raise

# --- Pinecone 임베딩 생성 ---
def embed_text(text: str):
    result = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[text],
        parameters={"input_type": "passage", "truncate": "END"}
    )
    print(result)
    return result['data'][0]['values']

# --- 검색 함수 ---
def query_pinecone(text: str, top_k: int = 2):
    embedding = embed_text(text)
    query_result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return query_result['matches']

@app.on_event("startup")
async def startup_event():
    global llm_model, tokenizer, pc, index, langchain_llm, llm_chain

    print("FastAPI 시작 - LLM 및 Pinecone 설정 중...")

    # LLM 로드
    llm_model, tokenizer = load_llm_and_tokenizer(BASE_MODEL_ID, LORA_ADAPTER_PATH, quantization_config)

    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
        raise ValueError("PINECONE 관련 환경변수가 누락되었습니다.")

        # LangChain용 LLM 래핑
    langchain_llm = wrap_llm_with_langchain(llm_model, tokenizer)

    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_template("""
        당신은 질문에 대해 주어진 컨텍스트를 바탕으로 답변하는 AI 어시스턴트입니다.
        컨텍스트를 사용하여 사용자의 질문에 최대한 상세하고 친절하게 한국어로 답변해주세요.
        컨텍스트에서 답을 찾을 수 없다면, "컨텍스트에서 관련된 정보를 찾을 수 없습니다."라고 답변해주세요.

        컨텍스트:
        {context}

        질문:
        {question}
    """)

    # 메모리 추가
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # LLMChain 생성
    llm_chain = LLMChain(
        llm=langchain_llm,
        prompt=prompt,
        memory=memory,
        verbose=True  # 디버깅용 로그 출력
    )

    # Pinecone 클라이언트 및 인덱스 초기화
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("llama-text-embed-v2-index")

    print("모든 초기화 완료.")

class RAGQueryRequest(BaseModel):
    query: str

class RAGQueryResponse(BaseModel):
    answer: str
    retrieved_context: list

@app.post("/query_rag", response_model=RAGQueryResponse)
async def query_rag_endpoint(request: RAGQueryRequest):
    if not llm_model or not tokenizer or not index or not llm_chain:
        raise HTTPException(status_code=503, detail="서버 초기화가 완료되지 않았습니다.")
    
    if not request.query:
        raise HTTPException(status_code=400, detail="질문이 비어있습니다.")

    try:
        print(f"[입력 쿼리] {request.query}")
        # 1. Pinecone에서 문서 검색
        matches = query_pinecone(request.query, top_k=2)
        print(type(matches), matches)
        retrieved_contexts = [
            # {"page_content": match['metadata'].get('content', ''), "metadata": match['metadata']}
            {"page_content": match['metadata'].get('content', ''), "metadata": match['metadata']}
            for match in matches
        ]

        # 2. 컨텍스트 조합
        context_string = "\n\n".join([ctx["page_content"] for ctx in retrieved_contexts])

        # 3. LangChain LLMChain에 전달할 입력 구성
        input_dict = {
            "context": context_string,
            "question": request.query
        }

        # 4. LangChain을 통한 응답 생성
        answer = llm_chain.run(input_dict)

        print(f"[모델 응답] {answer}")
        return RAGQueryResponse(answer=answer, retrieved_context=retrieved_contexts)

    except Exception as e:
        print(f"에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))
