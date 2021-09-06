import os


def process(split="dev"):
    file = os.path.join("/home/data_ti6_c/wangdq/data/ST/ende/" + split)
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
            audio = "/home/data_ti6_c/wangdq/data/ST/ende/fbank80.zip:" + ":".join(audio.split(":")[-2:])
            content.append("\t".join([sid, audio, n_frame, src, trg, speaker]))

    file = "/home/wangdq/ST/en-de/ST/" + split + '.tsv'
    with open(file, 'w') as fsrc:
        fsrc.write("\t".join(head) + '\n')
        fsrc.writelines(content)


def add_asr(split="dev"):
    file = os.path.join("/home/data_ti6_c/wangdq/data/ST/ende/ST/" + split + '.tsv')
    head = ['id', 'audio', 'n_frames', 'src_text', 'tgt_text', 'asr_output', 'speaker']
    content = []

    with open("/home/wangdq/test/generate-" + split + ".txt.hypo") as f:
        asr_output = f.readlines()

    with open(file) as f:
        for index, line in enumerate(f):
            if index == 0:
                continue

            line = line.split('\t')
            content.append('\t'.join(line[:-1] + [asr_output[index - 1].strip()] + [line[-1]]))

    with open('/home/wangdq/asr', 'w') as f:
        f.write('\t'.join(head) + '\n')
        f.writelines(content)


if __name__ == '__main__':
    add_asr("dev")
