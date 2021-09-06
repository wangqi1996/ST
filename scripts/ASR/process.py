
import os
def process(split="dev"):
    file = os.path.join("/home/data_ti6_c/wangdq/data/ST/ende/" + split)
    src_file = file + '_asr.tsv'
    head = ['id', 'audio', 'n_frames', 'tgt_text', 'speaker']
    content = []
    with open(src_file) as fsrc:
        for src in fsrc:
            sid, audio, n_frame, src, speaker = src.split('\t')
            if sid == "id":
                continue
            audio = "/home/data_ti6_c/wangdq/data/ST/ende/fbank80.zip:" + ":".join(audio.split(":")[-2:])
            content.append("\t".join([sid, audio, n_frame, src, speaker]))

    file = "/home/wangdq/ST/en-de/ASR/" + split + '.tsv'
    with open(file, 'w') as fsrc:
        fsrc.write("\t".join(head) + '\n')
        fsrc.writelines(content)


if __name__ == '__main__':
    process("dev")