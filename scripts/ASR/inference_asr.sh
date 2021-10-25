export CUDA_VISIBLE_DEVICES=$1

dirname=/home/data_ti6_c/wangdq/ST/external/ende/
fairseq-generate $dirname/ST \
  --config-yaml $dirname/ASR/asr.yaml --gen-subset train --task speech_to_text \
  --path $dirname/ASR/asr.pt \
  --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a \
  --results-path ~/$2 \
  --max-source-positions 6000 \
  --tgt-field src_text

#tail -1 ~/$2/generate-test.txt
#  --wer-lowercase --wer-remove-punct
