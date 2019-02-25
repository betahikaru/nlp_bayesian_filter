import unittest

from nlp_bayesian_filter.bayes import BayesianFilter


class TestAll(unittest.TestCase):
    def test_all(self):
        bf = BayesianFilter()
        bf.fit('今だけ三割引', '広告')
        bf.fit('美味しくなって再登場', '広告')
        bf.fit('打ち合わせよろしくお願いします', '重要')
        bf.fit('会議の議事録を送付いたします', '重要')

        predicted_category, score_list = bf.predict_category('激安！今だけ割引')
        print(f"結果={predicted_category}, list={score_list}")
