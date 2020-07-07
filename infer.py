import nemo
import nemo.collections.asr as nemo_asr
import numpy as np
import time
import soundfile as sf
import torch
from nemo.backends.pytorch.nm import DataLayerNM
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType
from ruamel.yaml import YAML
import os
from g2pNp2g_simple.p2gFuntion import p2g_simple as p2g
# simple data layer to pass audio signal
class AudioDataLayer(DataLayerNM):
    @property
    def output_ports(self):
        return {
            'audio_signal': NeuralType(('B', 'T'), AudioSignal(freq=self._sample_rate)),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

    def __init__(self, sample_rate):
        super().__init__()
        self._sample_rate = sample_rate
        self.output = True
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.output:
            raise StopIteration
        self.output = False
        return torch.as_tensor(self.signal, dtype=torch.float32), \
               torch.as_tensor(self.signal_shape, dtype=torch.int64)
        
    def set_signal(self, signal):
        self.signal = np.reshape(signal, [1, -1])
        self.signal_shape = np.expand_dims(self.signal.size, 0).astype(np.int64)
        self.output = True

    def __len__(self):
        return 1

    @property
    def dataset(self):
        return None

    @property
    def data_iterator(self):
        return self

    # Instantiate necessary neural modules

def __ctc_decoder_predictions_tensor(tensor, labels):
    blank_id = len(labels)
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    prediction_cpu_tensor = tensor.long().cpu()
    #print(prediction_cpu_tensor.shape)
    #for ind in range(prediction_cpu_tensor.shape[0]):
    prediction = prediction_cpu_tensor[0].numpy().tolist()
    # CTC decoding procedure
    decoded_prediction = []
    previous = len(labels)  # id of a blank symbol
    for p in prediction:
        p = np.argmax(p)
        #print(p)
        if (p != previous or previous == blank_id) and p != blank_id:
            decoded_prediction.append(p)
        previous = p
    hypothesis = '_'.join([labels_map[c] for c in decoded_prediction]).replace('_ _',' ').strip('_')
    hypotheses.append(hypothesis)
    return hypotheses

def load_audio(filename):
    with sf.SoundFile(filename, 'r') as f:
        sample_rate = f.samplerate
        samples = f.read()
        samples = samples.transpose()
    return samples

def restore_model(config_file, encoder_checkpoint, decoder_checkpoint):

    MODEL_YAML = config_file#'config/quartznet12x1.yaml'
    CHECKPOINT_ENCODER = encoder_checkpoint#'QuartzNet12x1_vivos/checkpoints/JasperEncoder-STEP-36700.pt'
    CHECKPOINT_DECODER = decoder_checkpoint#'QuartzNet12x1_vivos/checkpoints/JasperDecoderForCTC-STEP-36700.pt'

    yaml = YAML(typ="safe")
    with open(MODEL_YAML) as f:
        model_definition = yaml.load(f)

    # some changes for streaming scenario
    model_definition['AudioToMelSpectrogramPreprocessor']['dither'] = 0
    model_definition['AudioToMelSpectrogramPreprocessor']['pad_to'] = 0

    neural_factory = nemo.core.NeuralModuleFactory(
        placement=nemo.core.DeviceType.GPU,
        backend=nemo.core.Backend.PyTorch)

    data_layer = AudioDataLayer(sample_rate=model_definition['sample_rate'])

    data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(**model_definition['AudioToMelSpectrogramPreprocessor'])

    jasper_encoder = nemo_asr.JasperEncoder(
        feat_in=model_definition['AudioToMelSpectrogramPreprocessor']['features'],
        **model_definition['JasperEncoder'])

    jasper_decoder = nemo_asr.JasperDecoderForCTC(
        feat_in=model_definition['JasperEncoder']['jasper'][-1]['filters'],
        num_classes=len(model_definition['labels']))

    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    # load pre-trained model
    jasper_encoder.restore_from(CHECKPOINT_ENCODER)
    jasper_decoder.restore_from(CHECKPOINT_DECODER)

    # Define inference DAG
    audio_signal, audio_signal_len = data_layer()
    #print(audio_signal, audio_signal_len)
    #exit()
    processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)
    encoded, encoded_len = jasper_encoder(audio_signal=processed_signal, length=processed_signal_len)
    log_probs = jasper_decoder(encoder_output=encoded)
    predictions = greedy_decoder(log_probs=log_probs)

    # inference method for audio signal (single instance)
    def infer_signal(self, signal):
        data_layer.set_signal(signal)
        tensors = self.infer([log_probs], verbose=False)
        logits = tensors[0][0]
        #print('logits:',logits.shape)
        labels = list(model_definition['labels'])
        #print(type(labels))
        text = __ctc_decoder_predictions_tensor(logits, labels)[0]
        return text

    neural_factory.infer_signal = infer_signal.__get__(neural_factory)
    
    return neural_factory

# config = 'config/quartznet12x1.yaml'
# encoder_checkpoint = 'QuartzNet12x1_all/checkpoints/JasperEncoder-STEP-494800.pt'
# decoder_checkpoint = 'QuartzNet12x1_all/checkpoints/JasperDecoderForCTC-STEP-494800.pt'

# neural_factory = restore_model(config, encoder_checkpoint, decoder_checkpoint)
# print('restore model checkpoint done!')
# for f in os.listdir('audio_test'):
#     sig = load_audio('audio_test/' +f)
#     predicted = neural_factory.infer_signal(sig)#[0]#.cpu().numpy()[0]
#     print('predicted:', predicted)
#     print('predicted:', p2g(predicted))

