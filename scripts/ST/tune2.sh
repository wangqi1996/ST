export CUDA_VISIBLE_DEVICES=$1
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

dirname=/home/data_ti6_c/wangdq/ST/external/ende/
fairseq-train $dirname/mustc-ST \
  --config-yaml $dirname/mustc-ST/st.yaml --valid-subset dev_asr --train-subset train_asr \
  --save-dir ~/save/ST/tune_external/ \
  --num-workers 0 --max-tokens 10000 --update-freq 8 --max-update 100000 \
  --task speech_transcript_to_text --criterion st_loss --label-smoothing 0.1 \
  --arch st_transformer_adapter3 --optimizer adam --lr 6e-4 --lr-scheduler fixed \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --find-unused-parameters \
  --save-interval-updates 500 --keep-interval-updates 1 \
  --keep-best-checkpoints 10 --no-epoch-checkpoints \
  --log-interval 100 \
  --ASR-config $dirname/ASR/asr.yaml \
  --ASR-path $dirname/ASR/asr.pt \
  --MT-path $dirname/MT/mt.pt \
  --freeze-ASR --freeze-adapter \
  --mse-loss --MT-loss \
  --tgt-field tgt_text \
  --restore-file ~/save/ST/external2/checkpoint_ave_best.pt \
  --reset-optimizer --reset-dataloader --reset-meters --reset-meters
