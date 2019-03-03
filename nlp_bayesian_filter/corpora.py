import numpy as np


class Dictionary(object):
    def __init__(self):
        # key: word, value: word index
        self.word_dict = {}
        # index: document index, value: dict[key: word index, value: word occor count in the document]
        self.document_bag_of_words = []
        # key: word index, value: dict[key: "tf" or "idf", value: tf-value or idf-value]
        self.tf_idfs = {}

    def add_word(self, word):
        if word not in self.word_dict.keys():
            self.word_dict[word] = len(self.word_dict.keys())
            # print(f"word={word}, word index={self.word_dict[word]}")

    def get_words(self):
        return self.word_dict.keys()

    def add_document(self, word_list):
        bag_of_words = {}
        for word in word_list:
            self.add_word(word)
            word_index = self.word_dict[word]
            if word_index not in bag_of_words.keys():
                bag_of_words[word_index] = 0
            bag_of_words[word_index] += 1
        document_index = len(self.document_bag_of_words)
        self.document_bag_of_words.append(bag_of_words)
        print(f"document index={document_index}, bag of words={bag_of_words}")
        return document_index

    def get_document(self, document_index):
        bag_of_words = self.document_bag_of_words[document_index]
        return bag_of_words

    def get_idf_by_word(self, word):
        self.add_word(word)
        word_index = self.word_dict[word]
        return self.get_idf_by_word_index(word_index)

    def get_idf_by_word_index(self, word_index):
        # https://ja.wikipedia.org/wiki/Tf-idf
        document_count = len(self.document_bag_of_words)
        document_has_word_count = 10e-10
        for bag_of_words in self.document_bag_of_words:
            if word_index in bag_of_words.keys():
                document_has_word_count += 1
        idf = np.log(document_count/document_has_word_count) + 1
        return idf

    def get_tf_by_document_index_and_word(self, document_id, word):
        word_index = self.word_dict[word]
        return self.get_tf_by_document_index_and_word_index(word_index)

    def get_tf_by_document_index_and_word_index(self, document_id, word_index):
        # https://ja.wikipedia.org/wiki/Tf-idf
        all_word_count_in_document = 0
        specify_word_count_in_document = 0
        bag_of_words = self.document_bag_of_words[document_id]
        for a_word_index, word_count in bag_of_words.items():
            all_word_count_in_document += word_count
            if a_word_index == word_index:
                specify_word_count_in_document += word_count
        if specify_word_count_in_document == 0:
            return 0
        tf = specify_word_count_in_document / all_word_count_in_document
        return tf
