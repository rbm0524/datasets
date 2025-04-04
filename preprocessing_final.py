import os
import json
import re
from random import shuffle

def process_cornell(dir_path):
    lines_file = os.path.join(dir_path, 'movie_lines.txt')
    conv_file = os.path.join(dir_path, 'movie_conversations.txt')
    
    lines = {}
    with open(lines_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 5:
                lines[parts[0]] = parts[-1]
    
    data = []
    with open(conv_file, 'r', encoding='iso-8859-1') as f:
        for line in f:
            parts = line.strip().split(" +++$+++ ")
            if len(parts) == 4:
                utterance_ids = eval(parts[3])
                for i in range(len(utterance_ids) - 1):
                    u1, u2 = lines.get(utterance_ids[i]), lines.get(utterance_ids[i+1])
                    if u1 and u2:
                        data.append({
                            "instruction": u1.strip(),
                            "input": "",
                            "output": u2.strip()
                        })
    return data

def process_dailydialog(dir_path):
    file_path = os.path.join(dir_path, 'dialogues_text.txt')
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            utterances = [utt.strip() for utt in line.strip().split("__eou__") if utt.strip()]
            for i in range(len(utterances) - 1):
                data.append({
                    "instruction": utterances[i],
                    "input": "",
                    "output": utterances[i + 1]
                })
    return data

def process_opensubtitles(dir_path):
    data = []
    for file in os.listdir(dir_path):
        if file.endswith(".txt"):
            with open(os.path.join(dir_path, file), 'r', encoding='utf-8', errors='ignore') as f:
                lines = [line.strip() for line in f if line.strip()]
                for i in range(len(lines) - 1):
                    if 1 < len(lines[i]) < 100 and 1 < len(lines[i + 1]) < 100:
                        data.append({
                            "instruction": lines[i],
                            "input": "",
                            "output": lines[i + 1]
                        })
    return data

def process_santa_barbara(dir_path):
    data = []
    for file in os.listdir(dir_path):
        if file.endswith(".cha"):
            file_path = os.path.join(dir_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            speakers = {}  # 참가자 정보 저장
            dialogues = []  # 대화 저장

            for line in lines:
                line = line.strip()
                
                # 참가자 정보 추출 (@Participants: LENO LENORE Speaker, ...)
                if line.startswith("@Participants:"):
                    parts = line.split(":")[1].split(",")
                    for part in parts:
                        tokens = part.strip().split()
                        if len(tokens) >= 2:
                            code, name = tokens[0], tokens[1]
                            speakers[code] = name  # {LENO: LENORE, LYNN: LYNNE, ...}

                # 발화 추출 (*화자: 내용)
                match = re.match(r"^\*(\w+):\s*(.+)", line)
                if match:
                    speaker_code = match.group(1)
                    text = match.group(2)

                    # 화자 코드가 존재하는 경우만 추가
                    if speaker_code in speakers:
                        dialogues.append({"speaker": speakers[speaker_code], "text": text.strip()})

            # 대화를 instruction-output 형식으로 변환
            for i in range(len(dialogues) - 1):
                if dialogues[i]["speaker"] != dialogues[i + 1]["speaker"]:
                    data.append({
                        "instruction": dialogues[i]["text"],
                        "input": "",
                        "output": dialogues[i + 1]["text"]
                    })

    return data

def save_dataset(data, out_path):
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    cornell_data = process_cornell('./Cornell Movie-Dialogs Corpus')
    dailydialog_data = process_dailydialog('./DailyDialog/EMNLP_dataset')
    opensub_data = process_opensubtitles('./conversational-datasets')
    sbc_data = process_santa_barbara('./Santa Barbara Corpus/transcripts')

    all_data = cornell_data + dailydialog_data + opensub_data + sbc_data
    shuffle(all_data)

    save_dataset(all_data, 'llama_dialog_dataset_final.jsonl')
    print(f"✅ 총 {len(all_data)}개 샘플 저장 완료: llama_dialog_dataset_final.jsonl")
