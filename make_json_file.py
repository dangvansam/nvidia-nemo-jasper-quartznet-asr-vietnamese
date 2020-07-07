import os
import json
import librosa

max_dur = 0
data_dir = "/media/trung/win10/DATA/dataset_ASR/thaison_bysam/voip_audio_cuted/"
with open("data/voip_audio_cuted_transcript_simple.json", "a", encoding="utf-8") as fout:
    with open("converted2phone_simple/voip_audio_cuted_transcript_simple.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line = line.split("|")
            wav = line[0]
            text = line[1]#.decode("utf-8") 
            print(wav,text)
            wav = data_dir + wav + '.wav'
            sig, sr = librosa.load(wav, sr=None)
            print(sr)
            dur = len(sig)/sr
            print(dur)
            if dur > max_dur:
                max_dur = dur

            fout.write(json.dumps({"audio_filepath":wav, "duration":dur, "text":text}, ensure_ascii=False)+'\n')
print(max_dur)


