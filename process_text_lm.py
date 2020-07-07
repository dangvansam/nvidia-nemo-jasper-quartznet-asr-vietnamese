import os
from random import shuffle

data_dir = 'converted2phone_simple'
out_file = open('out_text_phoneme.txt','w', encoding='utf-8')
text_data = []
for f in os.listdir(data_dir):
    if f .split('.')[-1] != 'txt':
        continue
    with open(data_dir+'/'+f, 'r', encoding='utf-8') as fi:
        for line in fi:
            line = line.strip().split('|')[1]
            print(line)
            if len(line.split(' ')) < 2:
                continue
            text_data.append(line)
            #out_file.write(line)
            #out_file.write('\n')
print(len(text_data))
print(text_data[0])
shuffle(text_data)
print(text_data[0])
for l in text_data:
    out_file.write(l)
    out_file.write('\n')
