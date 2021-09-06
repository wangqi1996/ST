
"""
input: /home/data_ti6_c/wangdq/data/ST/ende/dev_*

1. extract the src text and the trg text
2. prepare.sh
    vocab: src_vocab == ASR's vocab, trg_vocab == MT's vocab

    # SCRIPTS=/home/data_ti5_c/wangdq/code/mosesdecoder/scripts/
    # TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
    # DETOKENIZER=$SCRIPTS/tokenizer/detokenizer.perl
    # src=en
    # trg=de
    # for split in train dev test; do
    #     for lang in $src $trg; do
    #         perl $TOKENIZER -threads 8 -l $lang < $split.$lang > $split.$lang.tok
    #     done
    # done
    # perl $CLEAN -ratio 1.5 train.tok $src $trg train.$lang.clean 1 175

    src=en
    trg=de
    src_vocab=/home/data_ti6_c/wangdq/data/ST/ende/asr_vocab.model
    trg_vocab=/home/data_ti6_c/wangdq/data/ST/ende/st_vocab.model
    for split in train dev test; do
        python /home/data_ti5_c/wangdq/code/st/scripts/spm_encode.py --model $src_vocab --inputs $split.$src --outputs $split.$src.bpe  --output_format piece
        mv $split.$src.bpe $split.$src
        python /home/data_ti5_c/wangdq/code/st/scripts/spm_encode.py --model $trg_vocab --inputs $split.$trg --outputs $split.$trg.bpe  --output_format piece
        mv $split.$trg.bpe $split.$trg
    done


    fairseq-preprocess --source-lang en --target-lang de \
    --trainpref train --validpref dev --testpref test \
    --destdir data-bin/ \
    --workers 20 \
    --srcdict /home/data_ti6_c/wangdq/data/ST/ende/asr_vocab.txt \
    --tgtdict /home/data_ti6_c/wangdq/data/ST/ende/st_vocab.txt \





python /home/data_ti5_c/wangdq/code/st/scripts/spm_encode.py --model asr_vocab.model --inputs train.de --outputs $split.$lang.bpe  --output_format piece

"""
import os
def process(split="dev"):
    file = os.path.join("/home/data_ti6_c/wangdq/data/ST/ende/" + split)
    src_file = file + '_asr.tsv'
    trg_file = file + '_st.tsv'
    src_content, trg_content = [], []
    with open(src_file) as fsrc, open(trg_file) as ftrg:
        for src, trg in zip(fsrc, ftrg):
            sid, _, _, src, _ = src.split('\t')
            tid, _, _, trg, _ = trg.split('\t')
            if sid == "id":
                continue
            assert  sid == tid, sid + '\t' + tid
            src_content.append(src.strip() + '\n')
            trg_content.append(trg.strip() + '\n')

    src_file = "/home/wangdq/ST/en-de/" + split + '.en'
    trg_file = "/home/wangdq/ST/en-de/" + split + '.de'
    with open(src_file, 'w') as fsrc:
        fsrc.writelines(src_content)

    with open(trg_file, 'w') as ftrg:
        ftrg.writelines(trg_content)


if __name__ == '__main__':
    process("train")