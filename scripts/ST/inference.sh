export CUDA_VISIBLE_DEVICES=$1

gensubset=test_asr
dirname=/home/data_ti6_c/wangdq/ST/external/ende/

fairseq-generate $dirname/ST \
  --config-yaml $dirname/ST/st.yaml \
  --gen-subset $gensubset --task speech_transcript_to_text \
  --path ~/save/ST/$3/checkpoint_best.pt \
  --max-tokens 50000 --beam 5 \
  --tokenizer moses --scoring sacrebleu \
  --results-path ~/$2 \
  --max-source-positions 6000 \
  --gen-cls-name STGenerator2

tail -1 ~/$2/generate-$gensubset.txt

python scripts/average_checkpoints.py --input ~/save/ST/$3/checkpoint.best_ --output ~/save/ST/$3/checkpoint_ave_best.pt

fairseq-generate $dirname/ST/ \
  --config-yaml $dirname/ST/st.yaml \
  --gen-subset $gensubset --task speech_transcript_to_text \
  --path ~/save/ST/$3/checkpoint_ave_best.pt \
  --max-tokens 50000 --beam 5 \
  --tokenizer moses --scoring sacrebleu \
  --results-path ~/$2 \
  --max-source-positions 6000 \
  --gen-cls-name STGenerator2

tail -1 ~/$2/generate-$gensubset.txt
