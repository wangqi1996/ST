export CUDA_VISIBLE_DEVICES=$1
# small模型在sentencepiece没有tokenizer，
# 而其余的模型有
# 这个要和ASR保持一致。

# ASR inference
SCRIPTS=/home/data_ti5_c/wangdq/code/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

file=$2
subset=test
dirname=/home/data_ti6_c/wangdq/ST/small_external/ende/
fairseq-generate $dirname/ST \
  --config-yaml $dirname/ASR/asr.yaml \
  --gen-subset $subset --task speech_to_text \
  --path $dirname/ASR/asr.pt \
  --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a \
  --results-path ~/$file \
  --max-source-positions 6000 \
  --tgt-field src_text

tail -1 ~/$file/generate-$subset.txt
python scripts/sort.py ~/$file/generate-$subset.txt
hypo_file=~/$file/generate-$subset.txt.hypo
ref_file=~/$file/generate-$subset.txt.ref
echo "ASR bleu"
cat $hypo_file | sacrebleu $ref_file
tail -n +2 $dirname/ST/$subset.tsv | cut -f5 >$hypo_file.de

# generate data-bin for MT
#cat $hypo_file |
#  perl $NORM_PUNC en |
#  perl $REM_NON_PRINT_CHAR |
#  perl $TOKENIZER -threads 8 -a -l en >>$hypo_file.tok
#cat $hypo_file.de |
#  perl $NORM_PUNC de |
#  perl $REM_NON_PRINT_CHAR |
#  perl $TOKENIZER -threads 8 -a -l de >>$hypo_file.de.tok

#mv $hypo_file $hypo_file
#mv $hypo_file.de $hypo_file.de

#dirname=/home/data_ti6_c/wangdq/ST/external/ende/

src_vocab=$dirname/asr.model
trg_vocab=$dirname/st.model
python /home/data_ti5_c/wangdq/code/st/scripts/spm_encode.py --model $src_vocab --inputs $hypo_file --outputs $hypo_file.bpe --output_format piece
python /home/data_ti5_c/wangdq/code/st/scripts/spm_encode.py --model $trg_vocab --inputs $hypo_file.de --outputs $hypo_file.de.bpe --output_format piece

mv $hypo_file.bpe $hypo_file.en
mv $hypo_file.de.bpe $hypo_file.de
fairseq-preprocess --source-lang en --target-lang de \
  --testpref $hypo_file \
  --destdir ~/$file/ \
  --workers 20 \
  --srcdict $dirname/asr.txt \
  --tgtdict $dirname/st.txt

# MT
fairseq-generate ~/$file/ \
  --gen-subset test --seed 1234 --task translation --batch-size 128 --beam 5 --remove-bpe -s en -t de \
  --path $dirname/MT/mt.pt --max-len-a 1.2 --max-len-b 10 --results-path ~/$file-MT/ \
  --bpe sentencepiece --sentencepiece-model $trg_vocab \
  --tokenizer moses --scoring sacrebleu

tail -1 ~/$file-MT/generate-test.txt
bash /home/data_ti5_c/wangdq/code/st/scripts/MT/compute_bleu.sh ~/$file-MT/generate-test.txt
