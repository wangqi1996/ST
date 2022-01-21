export CUDA_VISIBLE_DEVICES=$1

dirname=/home/data_ti6_c/wangdq/ST/external/ende/
fairseq-train $dirname/WMT_MT \
  --arch transformer_256 --share-decoder-input-output-embed \
  --encoder-embed-dim 256 --decoder-embed-dim 256 \
  --encoder-attention-heads 4 --decoder-attention-heads 4 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --dropout 0.3 --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 8000 \
  --eval-bleu --num-workers 0 \
  --eval-bleu-args '{"beam": 5}' \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe sentencepiece \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --keep-best-checkpoints 5 \
  --no-epoch-checkpoints \
  -s en -t de \
  --save-interval-updates 500 --keep-interval-updates 1 \
  --save-dir /home/wangdq/save/small_external/ \
  --bpe sentencepiece --sentencepiece-model $dirname/st.model \
  --max-update 100000
