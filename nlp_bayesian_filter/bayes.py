import math
import sys

from janome.tokenizer import Tokenizer

from .corpora import Dictionary


class BayesianFilter(object):
    """
    Bayesian Filter
    """

    def __init__(self):
        # provide index for word and document, calculate tf-idfs
        self.dictionary = Dictionary()
        # key: cagetory, value: dict[key: word, value: word occur count in the category]
        self.cagegory_word_count_dict = {}
        # key: category, value: category occur count
        self.category_count_dict = {}

    def split(self, text):
        """
        Parameters
        ==========
        text: str

        Returns
        =======
        result: list
        """
        result = []
        tokenizer = Tokenizer()
        malist = tokenizer.tokenize(text)
        for word in malist:
            surface = word.surface
            baseform = word.base_form
            if (baseform == '') or (baseform == '*'):
                baseform = surface
            result.append(baseform)
        return result

    def increment_word(self, word, category):
        if category not in self.cagegory_word_count_dict.keys():
            self.cagegory_word_count_dict[category] = {}
        if word not in self.cagegory_word_count_dict[category].keys():
            self.cagegory_word_count_dict[category][word] = 0
        self.cagegory_word_count_dict[category][word] += 1
        self.dictionary.add_word(word)

    def increment_category(self, category):
        if category not in self.category_count_dict.keys():
            self.category_count_dict[category] = 0
        self.category_count_dict[category] += 1

    def fit(self, text, category):
        word_list = self.split(text)
        for word in word_list:
            self.increment_word(word, category)
        self.increment_category(category)
        self.dictionary.add_document(word_list)

    def category_score(self, words, category):
        category_prob_v = self.category_prob(category)
        score = math.log(category_prob_v)
        for word in words:
            word_prob_v, word_prob_v2 = self.word_prob(word, category)
            print(f"category={category}, word={word}, word_prob_v={word_prob_v}, word_prob_v2={word_prob_v2}")
            score += math.log(word_prob_v2)
        return score

    def predict_category(self, text):
        best_category = None
        max_score = -sys.maxsize
        words = self.split(text)
        score_list = []
        for category in self.category_count_dict.keys():
            score = self.category_score(words, category)
            score_list.append((category, score))
            if score > max_score:
                max_score = score
                best_category = category
        return best_category, score_list

    def get_word_count_in_category(self, word, category):
        if category not in self.cagegory_word_count_dict.keys():
            raise Exception(f"category({category}) is not found.")
        if word in self.cagegory_word_count_dict[category]:
            return self.cagegory_word_count_dict[category][word]
        else:
            return 0

    def category_prob(self, category):
        """
        all words count in category / all words count in all category
        """
        sum_categories = sum(self.category_count_dict.values())
        category_v = self.category_count_dict[category]
        return category_v / sum_categories

    def word_prob(self, word, category):
        """
        specify word count in category / specify word count in all category
        """
        word_v = self.get_word_count_in_category(word, category) + 1  # math.log(0)を防ぐ
        sum_words = sum(self.cagegory_word_count_dict[category].values()) + len(self.dictionary.get_words())

        word_v2 = word_v * self.dictionary.get_idf_by_word(word)
        sum_words2 = 0
        for a_word, count in self.cagegory_word_count_dict[category].items():
            sum_words2 += count * self.dictionary.get_idf_by_word(a_word)
        return word_v / sum_words, word_v2 / sum_words2
