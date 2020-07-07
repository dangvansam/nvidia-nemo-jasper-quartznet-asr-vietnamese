import numpy as np
from flask import Flask, request, render_template, url_for
import argparse
import os
import sys
import numpy as np
import timeit
import scipy.io.wavfile as wav_
from infer import restore_model, load_audio
from g2pNp2g_simple.p2gFuntion import p2g_simple as p2g

app = Flask(__name__)

config = 'config/quartznet12x1.yaml'
encoder_checkpoint = 'QuartzNet12x1_all/checkpoints/JasperEncoder-STEP-494800.pt'
decoder_checkpoint = 'QuartzNet12x1_all/checkpoints/JasperDecoderForCTC-STEP-494800.pt'

neural_factory = restore_model(config, encoder_checkpoint, decoder_checkpoint)
print('restore model checkpoint done!')

@app.route('/')
def upload_form():
	return render_template('upload.html', audio_path = 'select file to predict!')

@app.route('/', methods=['POST'])
def get_prediction():
    print('PREDICT MODE')
    if request.method == 'POST':
        _file = request.files['file']
        if _file.filename == '':
            return upload_form()
        print('\n\nfile uploaded:',_file.filename)
        _file.save(os.path.join('static/uploaded', _file.filename))
        print('Write file success!')
        start = timeit.default_timer()
        sig = load_audio(os.path.join('static/uploaded', _file.filename))
        end_load = timeit.default_timer()
        predict = neural_factory.infer_signal(sig)
        predict_grapheme = p2g(predict)
        end_predict = timeit.default_timer()
        print('Load audio time:',end_load-start)
        print('Predict time:',end_predict-end_load)
        print('PRED Phoneme:{}'.format(predict))
        print('PRED Grapheme:{}'.format(predict_grapheme))

        return render_template('upload.html', predict_phoneme = predict, predict=predict_grapheme, predict_time=(end_predict-end_load), audio_path=os.path.join('static/uploaded', _file.filename))

if __name__ == '__main__':
    #load_model()  # load model at the beginning once only
    app.secret_key = 'dangvansam'
    app.run(host='192.168.2.26', port=4043, debug=True)
