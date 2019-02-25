import math
import sys

from janome.tokenizer import Tokenizer


class BayesianFilter(object):
    """
    Bayesian Filter
    """

    def __init__(self):
        self.words = set()
        self.word_dict = {}
        self.category_dict = {}

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
        if category not in self.word_dict.keys():
            self.word_dict[category] = {}
        if word not in self.word_dict[category].keys():
            self.word_dict[category][word] = 0
        self.word_dict[category][word] += 1
        self.words.add(word)

    def increment_category(self, category):
        if category not in self.category_dict.keys():
            self.category_dict[category] = 0
        self.category_dict[category] += 1

    def fit(self, text, category):
        word_list = self.split(text)
        for word in word_list:
            self.increment_word(word, category)
        self.increment_category(category)

    def category_score(self, words, category):
        category_prob_v = self.category_prob(category)
        score = math.log(category_prob_v)
        for word in words:
            word_prob_v = self.word_prob(word, category)
            print(f"word_prob_v={word_prob_v}")
            score += math.log(word_prob_v)
        return score

    def predict_category(self, text):
        best_category = None
        max_score = -sys.maxsize
        words = self.split(text)
        score_list = []
        for category in self.category_dict.keys():
            score = self.category_score(words, category)
            score_list.append((category, score))
            if score > max_score:
                max_score = score
                best_category = category
        return best_category, score_list

    def get_word_count_in_category(self, word, category):
        if category not in self.word_dict.keys():
            raise Exception(f"category({category}) is not found.")
        if word in self.word_dict[category]:
            return self.word_dict[category][word]
        else:
            return 0

    def category_prob(self, category):
        """
        all words count in category / all words count in all category
        """
        sum_categories = sum(self.category_dict.values())
        category_v = self.category_dict[category]
        return category_v / sum_categories

    def word_prob(self, word, category):
        """
        specify word count in category / specify word count in all category
        """
        word_v = self.get_word_count_in_category(word, category) + 1  # math.log(0)を防ぐ
        sum_words = sum(self.word_dict[category].values()) + len(self.words)
        return word_v / sum_words
