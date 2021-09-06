#MT:
#  best: 31.67  ave_best: 32.02
export CUDA_VISIBLE_DEVICES=$1

python scripts/average_checkpoints.py --input /home/wangdq/save/st/checkpoint.best_bleu --output /home/wangdq/save/st/checkpoint_ave_best.pt

fairseq-generate /home/data_ti6_c/wangdq/data/ST/ende/MT-data-bin/ \
  --gen-subset $3 \
  --seed 1234 \
  --task translation \
  --batch-size 128 \
  --beam 5 \
  --remove-bpe \
  -s en -t de \
  --path /home/wangdq/save/st/checkpoint_ave_best.pt \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --results-path ~/$2 \
  --model-overrides "{'valid_subset': '$3'}" \
  --bpe sentencepiece --sentencepiece-model /home/data_ti6_c/wangdq/data/ST/ende/st_vocab.model \
  --tokenizer moses \
  --scoring sacrebleu

tail -1 ~/$2/generate-$3.txt
