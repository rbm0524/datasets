# rag_llm_fastapi_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os
import pinecone
from dotenv import load_dotenv

# Langchain imports
from langchain_community.vectorstores import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings # 또는 다른 임베딩 모델
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline # 로컬 LLM용
from transformers import pipeline as hf_transformers_pipeline

# --- 설정 ---
load_dotenv() # .env 파일에서 환경 변수 로드

# LLM 설정 (이전과 유사)
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "MLP-KTLim/llama-3-Korean-Bllossom-8B")
LORA_ADAPTER_PATH = os.getenv("LORA_ADAPTER_PATH", "./lora-llama-3-korean-results_2") # 실제 경로로 수정
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Pinecone 설정
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# 임베딩 모델 설정
EMBEDDING_MODEL_NAME = "llama-text-embed-v2"

app = FastAPI()

# --- 전역 변수 (FastAPI 시작 시 초기화) ---
llm_model = None
tokenizer = None
hf_pipeline = None # HuggingFace Pipeline for Langchain
vector_store = None
retriever = None
rag_chain = None

# --- 모델 로드 함수 (이전 코드와 유사) ---
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
        raise # 시작 시 오류 발생 시 앱 중단

@app.on_event("startup")
async def startup_event():
    global llm_model, tokenizer, hf_pipeline, vector_store, retriever, rag_chain

    print("FastAPI 시작 - RAG 파이프라인 구성 요소 로드 중...")

    # LLM 및 토크나이저 로드
    llm_model, tokenizer = load_llm_and_tokenizer(BASE_MODEL_ID, LORA_ADAPTER_PATH, quantization_config)
    
    # Langchain용 HuggingFacePipeline 설정
    # Llama-3-Instruct 모델의 경우 tokenizer.chat_template 활용이 중요.
    # HuggingFacePipeline은 text-generation 파이프라인을 감싸며, 이 파이프라인이 chat_template을 잘 쓰도록 해야함.
    # 아래는 기본적인 text-generation 파이프라인. 필요시 custom task나 prompt 형식을 조정해야 할 수 있음.
    pipe = hf_transformers_pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        max_new_tokens=1024, # LLM 답변 최대 길이
        temperature=0.7, # 필요시 설정
        top_p=0.9,       # 필요시 설정
        # model_kwargs={"use_cache": True} # 필요시 설정
    )
    hf_pipeline = HuggingFacePipeline(pipeline=pipe)


    # 2. 임베딩 모델 로드
    print(f"임베딩 모델 로드 중: {EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': device})

    # 3. Pinecone 벡터 저장소 및 리트리버 설정
    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
        print("Pinecone 환경 변수가 설정되지 않았습니다. Pinecone 초기화 실패.")
        raise ValueError("PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME 환경변수를 설정해야 합니다.")

    print(f"Pinecone 초기화 중 (인덱스: {PINECONE_INDEX_NAME})...")
    # from pinecone import Pinecone as PineconeClient # 최신 Pinecone 클라이언트 사용법
    # pc = PineconeClient(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    # pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    # vector_store = Pinecone(pinecone_index, embeddings.embed_query, "text") # Pinecone v3.x용 Langchain 연동 방식

    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    vector_store = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings) # 기존 인덱스 사용
    
    retriever = vector_store.as_retriever(search_kwargs={'k': 2}) # 상위 2개 문서 검색

    # 4. RAG 프롬프트 템플릿 정의
    template = """
    당신은 질문에 대해 주어진 컨텍스트를 바탕으로 답변하는 AI 어시스턴트입니다.
    컨텍스트를 사용하여 사용자의 질문에 최대한 상세하고 친절하게 한국어로 답변해주세요.
    컨텍스트에서 답을 찾을 수 없다면, "컨텍스트에서 관련된 정보를 찾을 수 없습니다."라고 답변해주세요.

    컨텍스트:
    {context}

    질문: {question}

    답변:
    """
    prompt = PromptTemplate.from_template(template)

    # 5. Langchain RAG 체인 구성 (LCEL 사용)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | hf_pipeline # 로컬 LLM 파이프라인
        | StrOutputParser()
    )
    print("RAG 파이프라인 구성 완료.")


class RAGQueryRequest(BaseModel):
    query: str

class RAGQueryResponse(BaseModel):
    answer: str
    retrieved_context: list # 선택 사항: 어떤 컨텍스트가 사용되었는지 확인용

@app.post("/query_rag", response_model=RAGQueryResponse)
async def query_rag_endpoint(request: RAGQueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG 체인이 초기화되지 않았습니다.")
    if not request.query:
        raise HTTPException(status_code=400, detail="질문(query)이 비어있습니다.")

    print(f"RAG 질의 수신: {request.query}")
    try:
        # 컨텍스트 검색
        docs = await retriever.aget_relevant_documents(request.query)
        retrieved_context_for_response = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

        # RAG 체인 실행
        answer = await rag_chain.ainvoke(request.query)
        print(f"RAG 답변 생성: {answer}")
        return RAGQueryResponse(answer=answer, retrieved_context=retrieved_context_for_response)
    except Exception as e:
        print(f"RAG 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"RAG 처리 중 서버 오류 발생: {str(e)}")

# 서버 실행 방법
# uvicorn rag_llm_fastapi_server:app --reload --port 8000