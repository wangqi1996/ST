export CUDA_VISIBLE_DEVICES=$1

dirname=/home/data_ti6_c/wangdq/ST/small_external/ende/ASR/

fairseq-train $dirname \
  --config-yaml $dirname/asr.yaml --train-subset train --valid-subset dev \
  --save-dir ~/save/ASR/asr --num-workers 0 --max-tokens 40000 --max-update 300000 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 2 \
  --log-interval 500 --save-interval-updates 500 --keep-interval-updates 5 --no-epoch-checkpoints \
  --keep-best-checkpoints 10
