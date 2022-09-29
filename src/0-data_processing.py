import os
import gzip

### https://www.geeksforgeeks.org/time-process_time-function-in-python/
from time import process_time

filename = 'hg19.fa.gz'
filepath = os.path.join(os.getcwd(), 'dataset/word2vec/'+filename)

### Chromosome list
chr_list = []
for i in range(22):
    chr_list.append(str(i+1))
chr_list += ['X', 'Y', 'M']

t1_start = process_time()

hg19 = gzip.open(filepath)
sequence = hg19.read()
hg19.close()

### decoding: bytes -> str
sequence = sequence.decode(encoding='utf-8')

### 염기 외 문자열 제거
sequence = sequence.replace('\n','')
for chr_name in chr_list:
    chr = '>chr'+chr_name
    sequence = sequence.replace(chr,'')

### 모든 서열 대문자로
sequence = sequence.upper()

### 프로세싱 확인
# print(sequence[:10])

t1_end = process_time()
print("Elapsed time of parsing hg19 sequence:", t1_end - t1_start)

### 새로 파일 저장
new_filename = 'hg19_processed.fa'
new_filepath = os.path.join(os.getcwd(), 'dataset/word2vec/'+new_filename)

t2_start = process_time()

hg19_processed = open(new_filepath, 'w')
hg19_processed.write(sequence)
hg19_processed.close()

t2_end = process_time()
print("Elapsed time of saving the processed hg19 sequence:", t2_end - t2_start)

