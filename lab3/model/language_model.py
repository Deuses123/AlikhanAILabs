import os
import re
import numpy as np


class LanguageModel:
    def __init__(self):
        self.corpus = None
        self.build()

    def build(self):
        self.load_data()
        self.initialize_bigram_transition_probabilities()
        self.initialize_unigram_distributions()
        self.corpus = re.sub("[^а-яёА-ЯЁ]", " ", self.corpus)

        tokens = self.corpus.lower().split()
        for token in tokens:
            self.update_unigram_distribution(token[0])
            for i in range(len(token) - 1):
                self.update_transition_probability(token[i], token[i + 1])

        self.log_M = np.log(self.M / self.M.sum(axis=1, keepdims=True))
        self.log_pi = np.log(self.pi / self.pi.sum())

    def initialize_bigram_transition_probabilities(self):
        self.M = np.ones((34, 34))  # Используйте 32, чтобы учесть все русские буквы

    def initialize_unigram_distributions(self):
        self.pi = np.zeros(34)  # Используйте 32 для учета всех русских букв

    def letter_to_index(self, letter):
        return ord(letter) - ord('а')

    def update_transition_probability(self, char1, char2):
        i = self.letter_to_index(char1)
        j = self.letter_to_index(char2)
        self.M[i, j] += 1

    def update_unigram_distribution(self, char):
        i = self.letter_to_index(char)
        self.pi[i] += 1

    def get_log_word_probability(self, word):
        first_letter_index = self.letter_to_index(word[0])
        log_unigram_prob = (
            self.log_pi[first_letter_index] if first_letter_index in range(32) else 0
        )

        log_bigram_prob = 0
        for i in range(len(word) - 1):
            starting_letter_index = self.letter_to_index(word[i])
            ending_letter_index = self.letter_to_index(word[i + 1])
            log_bigram_prob += (
                self.log_M[starting_letter_index][ending_letter_index]
                if (starting_letter_index in range(32))
                   and (ending_letter_index in range(32))
                else 0
            )

        return log_unigram_prob + log_bigram_prob

    def get_sentence_log_probability(self, sentence):
        tokens = sentence.split()
        log_sentence_prob = sum(
            [self.get_log_word_probability(token) for token in tokens]
        )
        return log_sentence_prob

    def load_data(self):
        _init_file_dir = os.path.dirname(__file__)
        if os.path.exists(os.path.join(_init_file_dir, "book.txt").encode("utf-8")):
            with open(os.path.join(_init_file_dir, "book.txt"), "r", encoding="utf-8") as file:
                self.corpus = file.read()
