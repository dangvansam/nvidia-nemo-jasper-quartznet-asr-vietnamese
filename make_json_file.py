import os
import json
import librosa

max_dur = 0
data_dir = "/media/trung/nvme0n1p4/dataset_ASR/vlsp2020/test-vlsp2019/" #thư mục chứa file wav
#file json
with open("data/grapheme/test-vlsp2019.json", "w", encoding="utf-8") as fout:
    #file text
    with open("text_ori/test-vlsp2019.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line = line.split(" ",1)
            wav = data_dir + line[0] + ".wav"
            text = line[1]
            print(wav,text)
            sig, sr = librosa.load(wav, sr=None)
            print(sr)
            dur = len(sig)/sr
            print(dur)
            if dur > max_dur:
                max_dur = dur
            fout.write(json.dumps({"audio_filepath":wav, "duration":dur, "text":text}, ensure_ascii=False)+'\n')

print(max_dur)


