import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from transformers import Trainer, DataCollatorForSeq2Seq

# LoRA 설정
lora_config = LoraConfig(
    r=8,  # LoRA의 랭크 (값을 키우면 성능 증가 가능)
    lora_alpha=32,  # LoRA Scaling Factor
    target_modules=["q_proj", "v_proj"],  # LLaMA의 LoRA 적용 대상 모듈
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# 모델 및 토크나이저 로드
model_name = "meta-llama/Llama-8b-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,  # 메모리 최적화를 위해 8비트 사용
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = get_peft_model(model, lora_config)  # LoRA 적용

# 데이터 로드
dataset = load_dataset("json", data_files="llama_dialog_dataset_2.jsonl", split="train")

# 데이터 전처리
def preprocess_function(examples):
    instruction = examples["instruction"]
    output = examples["output"]
    text = f"### 질문: {instruction}\n### 답변: {output}"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512) # 실행 시켜보면서 조정 필요

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 데이터 콜레이터 설정
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 트레이닝 설정 - 파라미터도 실행시켜보며 조정 필요
training_args = TrainingArguments(
    output_dir="./lora-llama8b",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    push_to_hub=False
)

# Trainer 정의 및 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("lora-llama8b-checkpoint")
tokenizer.save_pretrained("lora-llama8b-checkpoint")