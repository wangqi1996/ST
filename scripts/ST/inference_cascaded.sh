export CUDA_VISIBLE_DEVICES=$1

# ASR inference
file=$2
fairseq-generate /home/data_ti6_c/wangdq/data/ST/ende/ST \
  --config-yaml /home/data_ti5_c/wangdq/code/st/config/ende/asr.yaml \
  --gen-subset test --task speech_to_text \
  --path /home/data_ti6_c/wangdq/model/ST/ende/ASR.pt \
  --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a \
  --results-path ~/$file \
  --max-source-positions 6000 \
  --tgt-field src_text

tail -1 ~/$file/generate-test.txt
python scripts/sort.py ~/$file/generate-test.txt
hypo_file=~/$file/generate-test.txt.hypo
ref_file=~/$file/generate-test.txt.ref
cat $hypo_file | sacrebleu $ref_file


# generate data-bin for MT
src_vocab=/home/data_ti6_c/wangdq/data/ST/ende/asr_vocab.model
trg_vocab=/home/data_ti6_c/wangdq/data/ST/ende/st_vocab.model
tail -n +2 /home/data_ti6_c/wangdq/data/ST/ende/ST/test.tsv | cut -f5  > $hypo_file.de
python /home/data_ti5_c/wangdq/code/st/scripts/spm_encode.py --model $src_vocab --inputs $hypo_file --outputs $hypo_file.bpe --output_format piece
python /home/data_ti5_c/wangdq/code/st/scripts/spm_encode.py --model $trg_vocab --inputs $hypo_file.de --outputs $hypo_file.de.bpe --output_format piece


mv $hypo_file.bpe $hypo_file.en
mv $hypo_file.de.bpe $hypo_file.de
fairseq-preprocess --source-lang en --target-lang de \
  --testpref $hypo_file \
  --destdir ~/$file/ \
  --workers 20 \
  --srcdict /home/data_ti6_c/wangdq/data/ST/ende/asr_vocab.txt \
  --tgtdict /home/data_ti6_c/wangdq/data/ST/ende/st_vocab.txt \

# MT
fairseq-generate /home/wangdq/$file/ \
  --gen-subset test --seed 1234 --task translation --batch-size 128 --beam 5 --remove-bpe -s en -t de \
  --path /home/wangdq/save/st/checkpoint_ave_best.pt --max-len-a 1.2 --max-len-b 10 --results-path ~/$file-MT/ \
  --bpe sentencepiece --sentencepiece-model /home/data_ti6_c/wangdq/data/ST/ende/st_vocab.model \
  --tokenizer moses --scoring sacrebleu

tail -1 ~/$file-MT/generate-test.txt