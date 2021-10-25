export CUDA_VISIBLE_DEVICES=$1
#DATA=/home/data_ti6_c/wangdq/data/ST/ende_Librispeech/data-bin
DATA=/home/data_ti6_c/wangdq/data/ST/large/MT/test/
DATA=/home/data_ti6_c/wangdq/data/ST/ende/MT-data-bin/
sentence_model=/home/data_ti6_c/wangdq/data/ST/large/MT/spm_unigram8000_st.model
sentence_model=/home/data_ti6_c/wangdq/data/ST/ende/st_vocab.model

fairseq-generate $DATA \
  --gen-subset $4 \
  --seed 1234 \
  --task translation \
  --batch-size 128 \
  --beam 5 \
  --remove-bpe \
  -s en -t de \
  --path /home/wangdq/save/$3/checkpoint_best.pt \
  --results-path ~/$2 \
  --model-overrides "{'valid_subset': '$4'}" \
  --bpe sentencepiece --sentencepiece-model $sentence_model \
  --tokenizer moses \
  --scoring sacrebleu

bash /home/data_ti5_c/wangdq/code/st/scripts/MT/compute_bleu.sh ~/$2/generate-$4.txt

#tail -1 ~/$2/generate-$4.txt
#
#python scripts/average_checkpoints.py --input /home/wangdq/save/$3/checkpoint.best_bleu --output /home/wangdq/save/$3/checkpoint_ave_best.pt
#
#fairseq-generate $DATA \
#  --gen-subset $4 \
#  --seed 1234 \
#  --task translation \
#  --batch-size 128 \
#  --beam 5 \
#  --remove-bpe \
#  -s en -t de \
#  --path /home/wangdq/save/$3/checkpoint_ave_best.pt \
#  --max-len-a 1.2 \
#  --max-len-b 10 \
#  --results-path ~/$2 \
#  --model-overrides "{'valid_subset': '$4'}" \
#  --bpe sentencepiece --sentencepiece-model /home/data_ti6_c/wangdq/data/ST/large/MT/spm_unigram8000_st.model \
#  --tokenizer moses \
#  --scoring sacrebleu
#
#tail -1 ~/$2/generate-$4.txt
