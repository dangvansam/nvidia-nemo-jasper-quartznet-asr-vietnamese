from ruamel.yaml import YAML
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.helpers import monitor_asr_train_progress, process_evaluation_batch, process_evaluation_epoch
from functools import partial

# các file json dùng để train
train_dataset = "/workspace/nemo_asr/json/vlsp2020_train_set_01.json"
train_dataset += ",/workspace/nemo_asr/json/vlsp2020_train_set_02.json"
train_dataset +=",/workspace/nemo_asr/json/vivos_train.json"
train_dataset += ",/workspace/nemo_asr/json/data_ctv.json"
train_dataset += ",/workspace/nemo_asr/json/fpt_open_set001_train_clean.json"
train_dataset += ",/workspace/nemo_asr/json/fpt_open_set001_test_clean.json"
train_dataset += ",/workspace/nemo_asr/json/wavenet.json"

# các file json dùng để valid
eval_datasets = "/workspace/nemo_asr/json/vivos_test.json"
eval_datasets += ",/workspace/nemo_asr/json/VLSP2020-T1-Transcript.json"
eval_datasets += ",/workspace/nemo_asr/json/VLSP2020-T2-Transcript.json"
eval_datasets += ",/workspace/nemo_asr/json/test-vc-vlsp18.json"

# QuartzNet Model definition
# Here we will be using separable convolutions
# with 12 blocks (k=12 repeated once r=1 from the picture above)
# chọn tham số mạng theo file config trong thư mục config
yaml = YAML(typ="safe")
with open("config/quartznet12x1_abcfjwz.yaml") as f:
    quartznet_model_definition = yaml.load(f)

log_dir = quartznet_model_definition["model"] + "_all_data"
nf = nemo.core.NeuralModuleFactory(log_dir=log_dir, placement=nemo.core.DeviceType.GPU, create_tb_writer=True)
tb_writer = nf.tb_writer

labels = quartznet_model_definition['labels']
print(len(labels), labels)
# Instantiate neural modules
data_layer = nemo_asr.AudioToTextDataLayer(manifest_filepath=train_dataset, sample_rate=16000, labels=labels, batch_size=32\
                                        ,shuffle=True, max_duration=20, trim_silence=False, normalize_transcripts=False)

data_layer_val = nemo_asr.AudioToTextDataLayer(manifest_filepath=eval_datasets, sample_rate=16000, labels=labels, batch_size=32\
                                        ,shuffle=False, max_duration=20, trim_silence=False, normalize_transcripts=False)

data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(**quartznet_model_definition['AudioToMelSpectrogramPreprocessor'])
spec_augment = nemo_asr.SpectrogramAugmentation(**quartznet_model_definition['SpectrogramAugmentation'])
encoder = nemo_asr.JasperEncoder(feat_in=quartznet_model_definition['AudioToMelSpectrogramPreprocessor']['features'], **quartznet_model_definition['JasperEncoder'])
decoder = nemo_asr.JasperDecoderForCTC(feat_in=1024, num_classes=len(labels))
ctc_loss = nemo_asr.CTCLossNM(num_classes=len(labels))
greedy_decoder = nemo_asr.GreedyCTCDecoder()

CHECKPOINT_ENCODER = 'quartznet12x1_12042021_finetune_from_15012021/checkpoints/JasperEncoder-STEP-330000.pt'
CHECKPOINT_DECODER = 'quartznet12x1_12042021_finetune_from_15012021/checkpoints/JasperDecoderForCTC-STEP-330000.pt'

encoder.restore_from(CHECKPOINT_ENCODER)
decoder.restore_from(CHECKPOINT_DECODER)

audio_signal, audio_signal_len, transcript, transcript_len = data_layer()
processed_signal, processed_signal_len = data_preprocessor(input_signal=audio_signal, length=audio_signal_len)

# Data argument
aug_signal = spec_augment(input_spec=processed_signal)
encoded, encoded_len = encoder(audio_signal=aug_signal, length=processed_signal_len)
log_probs = decoder(encoder_output=encoded)
predictions = greedy_decoder(log_probs=log_probs)
loss = ctc_loss(log_probs=log_probs, targets=transcript, input_length=encoded_len, target_length=transcript_len)

audio_signal_v, audio_signal_len_v, transcript_v, transcript_len_v = data_layer_val()
processed_signal_v, processed_signal_len_v = data_preprocessor(input_signal=audio_signal_v, length=audio_signal_len_v)
# Note that we are not using data-augmentation in validation DAG
encoded_v, encoded_len_v = encoder(audio_signal=processed_signal_v, length=processed_signal_len_v)
log_probs_v = decoder(encoder_output=encoded_v)
predictions_v = greedy_decoder(log_probs=log_probs_v)
loss_v = ctc_loss(log_probs=log_probs_v, targets=transcript_v, input_length=encoded_len_v, target_length=transcript_len_v)

train_callback = nemo.core.SimpleLossLoggerCallback(
    tb_writer=tb_writer,
    tensors=[loss, predictions, transcript, transcript_len],
    print_func=partial(monitor_asr_train_progress, labels=labels
    ))

saver_callback = nemo.core.CheckpointCallback(
    folder=log_dir+"/checkpoints", #load_from_folder="quartznet12x1_12042021_finetune_from_15012021/checkpoints", 
    step_freq=100, checkpoints_to_keep=1)

eval_callback = nemo.core.EvaluatorCallback(
    eval_tensors=[loss_v, predictions_v, transcript_v, transcript_len_v],
    user_iter_callback=partial(process_evaluation_batch, labels=labels),
    user_epochs_done_callback=partial(process_evaluation_epoch, tag="valid"),
    eval_step=1000,
    tb_writer=tb_writer,
    wandb_project="SAM-NEMO-0.10-ASR",
    wandb_name=log_dir
    )

wandb_callback = nemo.core.WandbCallback(
    train_tensors=[loss, predictions, transcript, transcript_len],
    wandb_project="SAM-NEMO-0.10-ASR",
    wandb_name=log_dir
    )

nf.train(
    tensors_to_optimize=[loss],
    callbacks=[train_callback, wandb_callback, eval_callback, saver_callback],
    optimizer="novograd",
    optimization_params={ "num_epochs": 150, "lr": 0.01, "weight_decay": 1e-4, "betas": [0.8, 0.5] }
    )