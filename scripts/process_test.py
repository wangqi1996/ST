

import os
import shutil
def process():

    key = 1000
    bin_dirname = "/home/data_ti5_c/wangdq/data/ema/data/split_docs/bin/" + str(key)
    raw_dirname = "/home/data_ti5_c/wangdq/data/ema/data/split_docs/"
    new_dirname = "/home/data_ti5_c/wangdq/data/ema/data/select/" + str(key)

    filelist = set(os.listdir(bin_dirname))
    os.makedirs(new_dirname)
    already = set()
    for f in os.listdir(raw_dirname):
        if f == 'bin' or not os.path.isdir(os.path.join(raw_dirname, f)):
            continue
        if int(f) > key + 31:
            continue
        for ff in os.listdir(os.path.join(raw_dirname, f)):
            if ff[:-3] in filelist:
                if ff in already:
                    raise AssertionError(ff)
                already.add(ff)
                shutil.copyfile(os.path.join(os.path.join(raw_dirname, f), ff), os.path.join(new_dirname, ff))



def check():
    dirname = "/home/data_ti5_c/wangdq/data/ema/data/select/1000/"
    # filename = "H-679-IV-de;H-717-Annex-de;lamictal_bi_de;menitorix_bi_de;Doxyprex_Background_Information-de;norfloxacin-bi-de;uman_big_q_a_de;sabumalin_q_a_de;etoricoxib-arcoxia-bi-de;sanohex_q_a_de;implanon_Q_A_de;emea-2006-0258-00-00-ende;V-121-de1;H-902-WQ_A-de;V-141-de1;V-137-de1;Hexavac-H-298-Z-28-de;H-891-de1;V-030-de1;V-107-de1;compagel-v-a-33-030-de;V-126-de1"
    # filename = "093604de1;H-741-de1;H-897-de1;H-933-de1;tritazide_q_a_de;V-133-de1;H-915-de1;H-725-de1;400803de1;H-890-de1;Veralipride-H-A-31-788-de;Belanette-AnnexI-III-de;V-A-35-029-de;112901de4"
    # filename = "49533907de;implanon_annexI_IV_de;EMEA-CVMP-82633-2007-de;sanohex_annexI_III_de;V-048-PI-de;V-041-PI-de;V-047-PI-de"
    # filename = "V-105-PI-de;H-391-PI-de;H-668-PI-de;H-960-PI-de"
    filename = "H-884-PI-de;H-287-PI-de;H-890-PI-de;H-273-PI-de;H-115-PI-de"
    filename = filename.split(';')
    filename.sort()
    en = []
    de = []
    for f in os.listdir(dirname):
        if f.endswith('.en'):
            en.append(f[:-3])
        elif f.endswith('de'):
            de.append(f[:-3])
    en.sort()
    de.sort()
    print(filename)
    print(de)
    print(en)
    print(len(filename), len(de), len(en))
    print("-----------")
    for f,e,d in zip(filename, en, de):
        if f != e or f != d:
            print(f, e,d)
    print("done")


if __name__ == '__main__':
    process()
    check()