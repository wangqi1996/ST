def process():
    dirname = "/home/data_ti6_c/wangdq/ST/external/ende/mustc-ST/"
    filename = dirname + "train_asr.tsv"
    tgt_contents = {}
    with open(filename, 'r') as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                continue
            line = line.strip().split('\t')
            tgt_contents[line[0]] = line[4]

    contents = {}
    new_content = []
    dirname2 = "/home/data_ti6_c/wangdq/ST/external/ende/ST/"
    filename = dirname2 + "train_asr.tsv"
    with open(filename, 'r') as f:
        for index, line in enumerate(f.readlines()):
            if index == 0:
                new_content.append(line)
                continue
            contents[line.strip().split('\t')[0]] = line

    _len = len(tgt_contents)
    import random
    samples = set(random.sample(list(tgt_contents.keys()), k=2000))

    for s in samples:
        c = contents[s]
        tgt = tgt_contents[s]
        c = c.strip().split('\t')
        c[4] = tgt
        c = "\t".join(c)
        new_content.append(c + '\n')

    with open("/home/wangdq/" + "train_2000.tsv", 'w') as f:
        f.writelines(new_content)


if __name__ == '__main__':
    process()
