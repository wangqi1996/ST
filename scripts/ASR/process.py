import os


def process(split="dev"):
    file = os.path.join("/home/data_ti6_c/wangdq/ST/Librispeech/" + split)
    src_file = file + '_asr.tsv'
    head = ['id', 'audio', 'n_frames', 'tgt_text', 'speaker']
    content = []
    with open(src_file) as fsrc:
        for index, src in enumerate(fsrc.readlines()):
            if index == 0:
                continue
            sid, audio, n_frame, _, src, speaker = src.strip().split('\t')
            if sid == "id":
                continue
            if "Librispeech" in audio:
                audio = "/home/data_ti6_c/wangdq/ST/Librispeech/fbank80.zip:" + ":".join(audio.split(":")[-2:])
            elif "mustc" in audio:
                audio = "/home/data_ti6_c/wangdq/ST/must-c/ende/fbank80.zip:" + ":".join(audio.split(":")[-2:])
            content.append("\t".join([sid, audio, n_frame, src, speaker]) + '\n')

    file = "/home/data_ti6_c/wangdq/ST/small_external/ende/ASR/" + split + '.tsv'
    with open(file, 'w') as fsrc:
        fsrc.write("\t".join(head) + '\n')
        fsrc.writelines(content)


if __name__ == '__main__':
    # process("dev")
    # process("test")
    process("train")
