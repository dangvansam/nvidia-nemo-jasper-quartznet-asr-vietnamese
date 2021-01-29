from infer import restore_model, load_audio
import os
from nemo.collections.asr.helpers import post_process_predictions, post_process_transcripts
from nemo.collections.asr.metrics import word_error_rate

#đường dẫn tới checkpoint và file config cho model
config = 'config/quartznet12x1_abcfjwz.yaml'
encoder_checkpoint = 'quartznet12x1_abcfjz_them100h/checkpoints/JasperEncoder-STEP-1312684.pt'
decoder_checkpoint = 'quartznet12x1_abcfjz_them100h/checkpoints/JasperDecoderForCTC-STEP-1312684.pt'

neural_factory = restore_model(config, encoder_checkpoint, decoder_checkpoint)
print('restore model checkpoint done!')

wav_dir = '/media/trung/nvme0n1p4//dataset_ASR/tts/'
text_file = '/media/trung/nvme0n1p4//dataset_ASR/tts_meta_data.txt'
f_text = open("result.txt","w", encoding="utf-8")
# f_error_file = open(error_file,"w")
with open(text_file, 'r') as f:
    for line in f:
        print("==============================")
        line = line.strip().split(" ",1)
        filename = line[0]+".wav"
        text = line[1]
        sig = load_audio(wav_dir+filename)
        _, beamlm = neural_factory.infer_signal(sig)
        if beamlm == "":
            f_text.write("{}{}\n".format(wav_dir,filename))
            continue
        if len(beamlm) != len(text):
            if len(beamlm) > len(text):
                text = text + " "*(len(beamlm)-len(text))
            else:
                beamlm = beamlm + " "*(len(text)-len(beamlm))
        wer = word_error_rate(hypotheses=beamlm, references=text, use_cer=False)*100
        # f_text.write("{}({}%)|{}|{}\n".format(filename,wer,text,beamlm))
        print("{}({}%)|{}|{}".format(filename,wer,text,beamlm))
        if wer >= 20:
            f_text.write("{}({}%)|{}|{}\n".format(filename,wer,text,beamlm))

f_text.close()
# f_error_file.close()