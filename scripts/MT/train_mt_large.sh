export CUDA_VISIBLE_DEVICES=$1

DISTILL=/home/data_ti6_c/wangdq/data/ST/ende_Librispeech/data-bin
fairseq-train $DISTILL \
  --arch transformer_wmt_en_de --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --dropout 0.3 --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 8000 \
  --fp16 --eval-bleu \
  --eval-bleu-args '{"beam": 5}' \
  --eval-bleu-detok moses --eval-bleu-remove-bpe sentencepiece \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --save-dir ~/save/ST/MT_large/ \
  --log-interval 500 --save-interval-updates 500 --keep-interval-updates 5 --no-epoch-checkpoints \
  --keep-best-checkpoints 5 \
  --num-workers 0 \
  -s en -t de \
  --bpe sentencepiece --sentencepiece-model /home/data_ti6_c/wangdq/data/ST/ende_Librispeech/spm_unigram8000_st.model \
  --max-update 300000
