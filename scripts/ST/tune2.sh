export CUDA_VISIBLE_DEVICES=$1
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

fairseq-train /home/data_ti6_c/wangdq/data/ST/ende/ST \
  --config-yaml /home/data_ti5_c/wangdq/code/st/config/ende/st.yaml --valid-subset dev_asr \
  --save-dir /home/wangdq/save/ST/adapter-asr-decoder/ \
  --num-workers 0 --max-tokens 20000 --update-freq 4 --max-update 300000 \
  --task speech_transcript_to_text --criterion st_loss --label-smoothing 0.1 \
  --arch pipelined_st --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --find-unused-parameters \
  --save-interval-updates 500 --keep-interval-updates 1 \
  --keep-best-checkpoints 5 --no-epoch-checkpoints \
  --log-interval 500 \
  --ASR-config /home/data_ti5_c/wangdq/code/st/config/ende/asr.yaml \
  --ASR-path /home/data_ti6_c/wangdq/model/ST/ende/ASR.pt \
  --MT-path /home/data_ti6_c/wangdq/model/ST/ende/MT_256_ave_2.pt \
  --freeze-ASR-encoder --freeze-NMT \
  --hidden-embedding-loss mse \
  --restore-file /home/data_ti6_c/wangdq/model/ST/ende/adapter.pt \
  --reset-optimizer --reset-dataloader --reset-meters --reset-meters
