from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import ne_chunk
from nltk import pos_tag


class Tokenizer:

    def __init__(self):
        self.data_sample = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."

    # 지정 부호 토크나이저 // 추후 진행
    def adj_tokenizer(self, data=None, split_point='.'):
        pre_ = data.strip().split(split_point)
        result = list(map(lambda i: i.strip(), pre_))

        if result[-1] == '':
            result.pop()
        return result

    def tokenizer(self, data=None):
        return word_tokenize(data)

    def pos_tag(self, data):
        return pos_tag(data)

    def ner(self, data):
        return ne_chunk(data)

    def stemming(self, token_list):
        stems = []
        ps = PorterStemmer()
        for token in token_list:
            stems.append(ps.stem(token))
        return stems

    def lemmatizer(self, token_list):
        lems = []
        wl = WordNetLemmatizer()
        for token in token_list:
            lems.append(wl.lemmatize(token))
        return lems

    def stop_words(self, postag, stop_pos=None, stop_word=None):
        if stop_pos is None:
            stop_pos = ['IN', 'CC', 'UH', 'TO', 'MD', 'DT', 'VBZ', 'VBP']
        if stop_word is None:
            stop_word = [',']

        words = []
        for tag in postag:
            if tag[1] not in stop_pos:
                if tag[0] not in stop_word:
                    words.append(tag[0])

        return words
