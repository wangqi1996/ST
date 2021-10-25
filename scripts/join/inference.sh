export CUDA_VISIBLE_DEVICES=$1

gensubset=test_asr
fairseq-generate /home/data_ti6_c/wangdq/data/ST/ende/ST \
  --config-yaml /home/data_ti5_c/wangdq/code/st/config/ende/st.yaml \
  --gen-subset $gensubset --task speech_transcript_to_text \
  --path /home/wangdq/save/join/$3/checkpoint_best.pt \
  --max-tokens 50000 --beam 5 \
  --tokenizer moses --scoring sacrebleu \
  --results-path ~/$2 \
  --max-source-positions 6000

tail -1 ~/$2/generate-$gensubset.txt

#python scripts/average_checkpoints.py --input /home/wangdq/save/join/$3/checkpoint.best_ --output /home/wangdq/save/join/$3/checkpoint_ave_best.pt
#
#fairseq-generate /home/data_ti6_c/wangdq/data/ST/ende/ST \
#  --config-yaml /home/data_ti5_c/wangdq/code/st/config/ende/st.yaml \
#  --gen-subset $gensubset --task speech_transcript_to_text \
#  --path /home/wangdq/save/join/$3/checkpoint_ave_best.pt \
#  --max-tokens 50000 --beam 5 \
#  --tokenizer moses --scoring sacrebleu \
#  --results-path ~/$2 \
#  --max-source-positions 6000
#
#tail -1 ~/$2/generate-$gensubset.txt
