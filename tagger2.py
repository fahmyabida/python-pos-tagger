from nltk.tag import CRFTagger

jumSample = 500000
namaFile = "Indonesian_Manually_Tagged_Corpus.tsv"
with open(namaFile, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

pasangan = []
allPasangan = []

for line in lines[: min(jumSample, len(lines))]:
    if line == '':
        allPasangan.append(pasangan)
        pasangan = []
    else:
        kata, tag = line.split('\t')
        p = (kata, tag)
        pasangan.append(p)

ct = CRFTagger()
ct.train(allPasangan, 'all_indo_man_tag_corpus_model.crf.tagger')
# test
hasil = ct.tag_sents([['Saya', 'bekerja', 'di', 'Bandung'], ['Nama', 'saya', 'Yudi']])
print(hasil)