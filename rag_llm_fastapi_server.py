from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
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
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False  # 입력 프롬프트 제외하고 생성된 텍스트만 반환
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
    return result.data[0]['values']

# --- 검색 함수 ---
def query_pinecone(text: str, top_k: int = 2):
    embedding = embed_text(text)
    query_result = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return query_result['matches'] # dictionary에서 'matches' 키를 통해 검색된 결과를 가져옴

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

    # Pinecone 클라이언트 및 인덱스 초기화
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("llama-text-embed-v2-index")
    
    # 메모리 추가
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=False)
    
    # 프롬프트 템플릿 생성 - 입력 변수 명확히 지정
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 일상 영어 회화 AI 어시스턴트입니다. 컨텍스트를 사용하여 사용자의 질문에 간단하게 답변해주세요. 컨텍스트에서 답을 찾을 수 없다면, 일반적인 지식을 사용하여 답변하세요."),
        ("human", "질문: {question}\n\n컨텍스트:\n{context}"),
    ])

    # LLMChain 생성 - 입력 키 명확히 지정
    llm_chain = LLMChain(
        llm=langchain_llm,
        prompt=prompt,
        memory=memory,  # 메모리 지정
        verbose=True,  # 디버깅용 로그 출력
        output_key="text"  # 출력 키 명확히 지정
    )

    print(f"LLMChain 입력 변수: {llm_chain.prompt.input_variables}")
    print("모든 초기화 완료.")

    print(f"LLMChain 입력 변수: {llm_chain.prompt.input_variables}")
    print("모든 초기화 완료.")

class RAGQueryRequest(BaseModel):
    query: str

class RAGQueryResponse(BaseModel):
    answer: str

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
        
        retrieved_contexts = [
            {"page_content": match['metadata'].get('content', ''), "metadata": match['metadata']}
            for match in matches
        ]

        # 2. 컨텍스트 조합 - 최대 길이 제한
        context_texts = [ctx["page_content"] for ctx in retrieved_contexts]
        # 컨텍스트가 너무 길다면, 앞부분만 사용
        max_context_length = 1000  # 적절한 길이로 조정
        context_string = "\n\n".join(context_texts)
        if len(context_string) > max_context_length:
            context_string = context_string[:max_context_length] + "..."

        # 3. LangChain LLMChain에 전달할 입력 구성 - 입력 키 맞춤
        chain_input = {
            "question": request.query,
            "context": context_string
        }
        
        print(f"LLMChain에 전달될 입력: {chain_input}")

        # 4. LangChain을 통한 응답 생성
        response = await llm_chain.ainvoke(chain_input)
        
        # 응답 추출 및 처리
        if 'text' in response:
            answer = response['text']
        else:
            # 응답 형식이 예상과 다를 경우 딕셔너리에서 문자열 값을 찾아 사용
            print(f"예상치 못한 응답 형식: {response}")
            for key, value in response.items():
                if isinstance(value, str) and len(value) > 0:
                    answer = value
                    break
            else:
                answer = "응답을 처리하는 동안 오류가 발생했습니다."
        
        # 응답 정리 (앞뒤 공백 제거)
        answer = answer.strip()
        
        print(f"[최종 모델 응답] {answer}")
        return RAGQueryResponse(answer=answer)

    except Exception as e:
        print(f"에러 발생: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        print(f"에러 발생: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        print(f"에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 실행 명령어:
# uvicorn rag_llm_fastapi_server:app --reload --port 8090
# 테스트 명령어: 
# curl -X POST http://127.0.0.1:8090/query_rag -H "Content-Type: application/json" -d "{\"query\": \"한국의 수도는 어디야?\"}"