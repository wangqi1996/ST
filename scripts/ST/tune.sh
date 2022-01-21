export CUDA_VISIBLE_DEVICES=$1
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

dirname=/home/data_ti6_c/wangdq/ST/small_external/ende/
fairseq-train $dirname/mustc-ST \
  --config-yaml $dirname/mustc-ST/st.yaml --valid-subset dev_asr --train-subset train_asr \
  --save-dir ~/save/ST/tune_small_external/ \
  --num-workers 0 --max-tokens 40000 --update-freq 2 --max-update 300000 \
  --task speech_transcript_to_text --criterion st_loss --label-smoothing 0.1 \
  --arch st_transformer_adapter2 --optimizer adam --lr 1e-3 --lr-scheduler fixed \
  --clip-norm 1.0 --seed 1 \
  --save-interval-updates 200 --keep-interval-updates 1 \
  --keep-best-checkpoints 10 --no-epoch-checkpoints --find-unused-parameters \
  --log-interval 200 \
  --ASR-config $dirname/ASR/asr.yaml \
  --ASR-path $dirname/ASR/asr.pt \
  --MT-path $dirname/MT/mt.pt \
  --freeze-ASR --freeze-adapter \
  --mse-loss --MT-loss \
  --tgt-field tgt_text \
  --restore-file /home/wangdq/save/ST/external/checkpoint_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters --reset-meters
