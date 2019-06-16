import string

stopwords_path = '/root/xcn/file4lab/esoda/datasets/stopwords.txt'
# test_folder = 'conf_chi'
test_folder = 'conf_ecscw'
test_file = 'conf_ecscw_BjornC11.txt'
path = '/root/xcn/file4lab/esoda/datasets/'


stopwords = []
with open(stopwords_path) as f:
    for word in iter(f):
        stopwords.append(word.strip())

input_text = ''
translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
with open(path + test_folder + '/' + test_file) as f:
    for line in iter(f):
        line = line.translate(translator)
        input_text += ' ' + line.strip()

from textRank import textRank
tr = textRank(input_text, stopwords)
tr.run()

