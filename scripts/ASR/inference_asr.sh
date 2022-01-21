export CUDA_VISIBLE_DEVICES=$1

dirname=/home/data_ti6_c/wangdq/ST/small_external/ende/
fairseq-generate $dirname/mustc-ST \
  --config-yaml $dirname/ASR/asr.yaml --gen-subset dev --task speech_to_text \
  --path $dirname/ASR/asr.pt \
  --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a \
  --results-path ~/$2 \
  --max-source-positions 6000 \
  --tgt-field src_text
#
#tail -1 ~/$2/generate-test.txt
#
#python scripts/average_checkpoints.py --input ~/save/ASR/asr/checkpoint.best_loss_2.33 --output ~/save/ASR/asr/checkpoint_ave_best.pt
#
#fairseq-generate $dirname/ASR \
#  --config-yaml $dirname/ASR/asr.yaml --gen-subset test --task speech_to_text \
#  --path ~/save/ASR/asr/checkpoint_ave_best.pt \
#  --max-tokens 50000 --beam 5 \
#  --scoring wer --wer-tokenizer 13a \
#  --results-path ~/$2 \
#  --max-source-positions 6000
#
#tail -1 ~/$2/generate-test.txt
