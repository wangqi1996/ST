import os


def process_li():
    dirname = "/home/data_ti6_c/wangdq/ST/Librispeech/"
    file = os.path.join(dirname + "train")
    src_file = file + '_asr.tsv'
    head = ['id', 'audio', 'n_frames', 'src_text', 'tgt_text', 'speaker']
    content = []
    with open(src_file) as fsrc:
        for line, src in enumerate(fsrc):
            if line == 0:
                continue
            sid, audio, n_frame, lang, src, speaker = src.strip().split('\t')
            trg = "none"
            # assert sid == tid, sid + '\t' + tid
            if "Librispeech" in audio:
                audio = dirname + "fbank80.zip:" + ":".join(audio.split(":")[-2:])
            elif "mustc" in audio:
                audio = "/home/data_ti6_c/wangdq/ST/must-c/ende/fbank80.zip:" + ":".join(audio.split(":")[-2:])
            content.append("\t".join([sid, audio, n_frame, src, trg, speaker]) + '\n')

    file = "/home/data_ti6_c/wangdq/ST/external/ende/ST/train.tsv"
    with open(file, 'w') as fsrc:
        fsrc.write("\t".join(head) + '\n')
        fsrc.writelines(content)


def process(split="dev"):
    dirname = "/home/data_ti6_c/wangdq/ST/Librispeech/"
    file = os.path.join(dirname + split)
    src_file = file + '_asr.tsv'
    trg_file = file + '_st.tsv'
    head = ['id', 'audio', 'n_frames', 'src_text', 'tgt_text', 'speaker']
    content = []
    with open(src_file) as fsrc, open(trg_file) as ftrg:
        for src, trg in zip(fsrc, ftrg):
            sid, audio, n_frame, src, speaker = src.split('\t')
            tid, _, _, trg, _ = trg.split('\t')
            if sid == "id":
                continue
            assert sid == tid, sid + '\t' + tid
            audio = dirname + "fbank80.zip:" + ":".join(audio.split(":")[-2:])
            content.append("\t".join([sid, audio, n_frame, src, trg, speaker]))

    file = "/home/data_ti6_c/wangdq/ST/external/ende/mustc-ST/" + split + '.tsv'
    with open(file, 'w') as fsrc:
        fsrc.write("\t".join(head) + '\n')
        fsrc.writelines(content)


def add_asr(split="dev"):
    hint = "external"
    dirname = "/home/data_ti6_c/wangdq/ST/" + hint + "/ende/ST/"
    file = os.path.join(dirname + split + '.tsv')
    head = ['id', 'audio', 'n_frames', 'src_text', 'tgt_text', 'asr_output', 'speaker']
    content = []

    with open("/home/wangdq/" + hint + "/generate-" + split + ".txt.hypo") as f:
        asr_output = f.readlines()

    with open(file) as f:
        for index, line in enumerate(f):
            if index == 0:
                continue

            line = line.split('\t')
            content.append('\t'.join(line[:-1] + [asr_output[index - 1].strip()] + [line[-1]]))

    file = "/home/wangdq/" +hint + "/" + split + '_asr.tsv'
    with open(file, 'w') as f:
        f.write('\t'.join(head) + '\n')
        f.writelines(content)


if __name__ == '__main__':
    add_asr("train")
    # process('train')
    # process('dev')
    # process('test')
    # process_li()

