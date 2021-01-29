import json
import re
import string
f_text = open("tts_meta_data.txt","w", encoding="utf-8")
with open("data/grapheme/tts_meta_data.json", "r", encoding="utf-8") as f:
    for line in f:
        a = json.loads(line)
        audio_path = a["audio_filepath"].split("/")[-1].split(".")[0]
        dur = a["duration"]
        text = a["text"]
        print(audio_path,text)
        f_text.write("{} {}\n".format(audio_path,text))
