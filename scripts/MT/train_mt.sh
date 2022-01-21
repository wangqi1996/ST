export CUDA_VISIBLE_DEVICES=$1
DATA=/home/data_ti6_c/wangdq/ST/middle/ende/MT
bpe_model=/home/data_ti6_c/wangdq/ST/small/ende/st.model

fairseq-train $DATA \
  --arch transformer_source_256 --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --dropout 0.3 --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4096 \
  --fp16 --eval-bleu \
  --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses --eval-bleu-remove-bpe "sentencepiece" \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --save-dir /home/data_ti6_d/wangdq/save/small_source/ \
  --log-interval 500 --save-interval-updates 500 --keep-interval-updates 5 --no-epoch-checkpoints \
  --keep-best-checkpoints 5 \
  --num-workers 0 \
  --bpe sentencepiece --sentencepiece-model $bpe_model \
  --reset-meters --max-update 150000 -s en -t de \
