def process():
    dirname = "/home/data_ti6_c/wangdq/data/ST/ende/ST/"
    filename = dirname + "train.tsv"
    with open(filename, 'r') as f:
        contents = f.readlines()

    _len = len(contents)
    import random
    samples = set(random.sample(range(_len), k=2000))
    new_content = []
    for id, line in enumerate(contents):
        if id in samples:
            new_content.append(line)
    with open(dirname + "train_2000.tsv", 'w') as f:
        f.writelines(new_content)


if __name__ == '__main__':
    process()
