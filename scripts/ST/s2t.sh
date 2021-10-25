export CUDA_VISIBLE_DEVICES=$1
export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'

fairseq-train /home/data_ti6_c/wangdq/data/ST/ende/ST \
  --config-yaml /home/data_ti5_c/wangdq/code/st/config/ende/st.yaml --valid-subset dev_asr \
  --save-dir /home/wangdq/save/ST/s2t/ \
  --num-workers 0 --max-tokens 10000 --max-update 100000 --update-freq 4 \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --arch s2t_adapter --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --find-unused-parameters \
  --save-interval-updates 500 --keep-interval-updates 1 \
  --keep-best-checkpoints 5 --no-epoch-checkpoints \
  --log-interval 500 \
  --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-args '{"beam": 5}' \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --ASR-config /home/data_ti5_c/wangdq/code/st/config/ende/asr.yaml \
  --load-pretrained-decoder-from /home/data_ti6_c/wangdq/model/ST/ende/MT.pt \
  --load-pretrained-encoder-from /home/data_ti6_c/wangdq/model/ST/ende/ASR.pt \
  --freeze-encoder --freeze-decoder
