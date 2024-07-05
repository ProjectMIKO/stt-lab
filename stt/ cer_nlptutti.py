import nlptutti as metrics

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

file1_path = 'ref.txt'
file2_path = 'stt.txt'

# 파일 내용 읽기
refs = read_file(file1_path) # References 기준이 되는 참조 텍스트 (정답)
preds = read_file(file2_path) # Predictions stt 텍스트

# CER 계산
result = metrics.get_cer(refs, preds)
cer = result['cer']
substitutions = result['substitutions']
deletions = result['deletions']
insertions = result['insertions']

print(result)
# 출력: [cer, substitutions, deletions, insertions] -> [CER = ..., S = ..., D = ..., I = ...]
