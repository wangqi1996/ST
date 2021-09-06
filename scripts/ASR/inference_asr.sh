export CUDA_VISIBLE_DEVICES=$1
fairseq-generate /home/data_ti6_c/wangdq/data/ST/ende/ST \
  --config-yaml /home/data_ti5_c/wangdq/code/st/config/ende/asr.yaml --gen-subset dev --task speech_to_text \
  --path /home/data_ti6_c/wangdq/model/ST/ende/ASR.pt \
  --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a \
  --results-path ~/$2 \
  --max-source-positions 6000 \
  --tgt-field src_text

tail -1 ~/$2/generate-test.txt
