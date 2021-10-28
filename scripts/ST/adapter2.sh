export CUDA_VISIBLE_DEVICES=$1
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

dirname=/home/data_ti6_c/wangdq/ST/small/ende/
fairseq-train $dirname/ST \
  --config-yaml $dirname/ST/st.yaml --valid-subset dev_asr --train-subset train_asr \
  --save-dir ~/save/ST/small_soft/ \
  --num-workers 0 --max-tokens 80000 --update-freq 2  --max-update 300000 \
  --task speech_transcript_to_text --criterion st_loss --label-smoothing 0.1 \
  --arch pipelined_st --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 \
  --save-interval-updates 500 --keep-interval-updates 1 \
  --keep-best-checkpoints 10 --no-epoch-checkpoints --find-unused-parameters \
  --log-interval 500 \
  --ASR-config $dirname/ASR/asr.yaml \
  --ASR-path $dirname/ASR/asr.pt \
  --MT-path $dirname/MT/mt.pt \
  --freeze-ASR --freeze-NMT \
  --mse-loss  \
  --tgt-field tgt_text \
  --adapter-input asr --encoder-input transcript \
  --predict-length --use-attention --attention-type soft

