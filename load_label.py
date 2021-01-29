import numpy as np
import json
#label = np.load("297phone_space.npy")
# label = " abcdeghiklmnopqrstuvxyàáâãèéêìíòóôõùúýăđĩũơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ"
# print(list(label))

# các file json dùng để train
train_dataset = "data/grapheme/vivos_train.json,data/grapheme/wavenet.json,data/grapheme/fpt_open_set001_train_clean.json"
train_dataset += ",data/grapheme/voip_audio_cuted_transcript.json,data/grapheme/to_tieng_transcript.json,data/grapheme/audio_18_11_cuted_transcript.json"
train_dataset += ",data/grapheme/part_01.json,data/grapheme/part_02_21102020.json" #data mới transcript
train_dataset += ",data/grapheme/vlsp2020_train_set_02.json" #data vin 100h
# các file json dùng để valid
eval_datasets = "data/grapheme/vivos_test.json,data/grapheme/fpt_open_set001_test_clean.json"
eval_datasets += ",data/grapheme/thaison_data_07012020_cuted.json"

json_files = train_dataset.split(",") + eval_datasets.split(",")
print(len(json_files))
word_vocab = []
with open("alltext.txt",'w',encoding='utf-8') as fout:
    for f in json_files:
        print(f)
        with open(f, 'r', encoding='utf-8') as fin:
            for line in fin:
                data = json.loads(line)
                #print(data)
                print(data["text"])
                fout.write(data["text"]+'\n')
                # for w in data["text"].split(" "):
                #     if w not in word_vocab:
                #         word_vocab.append(w)
    #exit()

# print(word_vocab)
# print(len(word_vocab))
