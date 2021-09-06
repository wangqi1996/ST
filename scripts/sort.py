def process(file):
    hypos, ref, ids = [], [], []
    with open(file) as f:
        for line in f:
            if line.startswith("D-"):
                hypos.append(line.split('\t')[-1])
                ids.append(int(line.split('\t')[0][2:]))
            elif line.startswith("T-"):
                ref.append(line.split('\t')[-1])

    sorted_hypos = [None for _ in ids]
    sorted_ref = [None for _ in ids]
    for index, id in enumerate(ids):
        sorted_ref[id] = ref[index]
        sorted_hypos[id] = hypos[index]

    with open(file + '.hypo', 'w') as f:
        f.writelines(sorted_hypos)

    with open(file + '.ref', 'w') as f:
        f.writelines(sorted_ref)


if __name__ == '__main__':
    import sys

    file = sys.argv[1]
    process(file)
