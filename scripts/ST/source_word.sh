export CUDA_VISIBLE_DEVICES=$1
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

fairseq-train /home/data_ti6_c/wangdq/data/ST/ende/ST \
  --config-yaml /home/data_ti5_c/wangdq/code/st/config/ende/st.yaml --valid-subset dev_asr \
  --save-dir /home/wangdq/save/ST/source_word/ \
  --num-workers 0 --max-tokens 20000 --max-update 100000 --update-freq 2 \
  --task speech_transcript_to_text --criterion st_loss --label-smoothing 0.1 \
  --arch pipelined_st --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --find-unused-parameters \
  --save-interval-updates 500 --keep-interval-updates 1 \
  --keep-best-checkpoints 5 --no-epoch-checkpoints \
  --log-interval 500 \
  --best-checkpoint-metric mse-0 \
  --ASR-config /home/data_ti5_c/wangdq/code/st/config/ende/asr.yaml \
  --ASR-path /home/data_ti6_c/wangdq/model/ST/ende/ASR.pt \
  --MT-path /home/data_ti6_c/wangdq/model/ST/ende/MT.pt \
  --reset-dataloader \
  --freeze-ASR --freeze-NMT \
  --hidden-embedding-loss mse \
  --source-word-loss
