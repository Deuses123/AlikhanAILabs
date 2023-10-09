import numpy as np
import random
from language_model import LanguageModel
from Encoder import Encoder
from copy import copy
import joblib


def load(name):
    return joblib.load(name)


def save(g):
    joblib.dump(g, 'model.pkl')


class GeneticAlgorithm:

    POOL_SIZE = 20
    OFFSPRING_POOL_SIZE = 5
    NUM_ITER = 250

    def __init__(self):
        self.__initialize_dna_pool()
        self.__initialize_encoder()
        self.__initialize_language_model()

    def __initialize_dna_pool(self):
        """Генерирует пул ДНК"""
        self.all_letters = list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        self.dna_pool = [
            "".join(list(np.random.permutation(self.all_letters)))
            for _ in range(self.POOL_SIZE)
        ]

        self.offspring_pool = []

    def __initialize_encoder(self):
        """Инициализирует объект Encoder как атрибут класса encoder"""
        self.encoder = Encoder()

    def __initialize_language_model(self):
        """Инициализирует объект LanguageModel как атрибут класса m"""
        self.m = LanguageModel()

    @staticmethod
    def random_swap(sequence):
        index_1, index_2 = random.sample(list(np.arange(len(sequence))), 2)
        sequence_copy = copy(sequence)
        seq_list = list(sequence_copy)
        seq_list[index_1], seq_list[index_2] = seq_list[index_2], seq_list[index_1]
        return "".join(seq_list)

    def evolve_offspring(self):
        for dna in self.dna_pool:
            self.offspring_pool += [
                self.random_swap(dna) for _ in range(self.OFFSPRING_POOL_SIZE)
            ]

        return self.offspring_pool + self.dna_pool

    def mutate(self, dna, mutation_rate):
        mutated_dna = list(dna)
        for i in range(len(dna)):
            if random.random() < mutation_rate:
                # Мутация: случайное изменение символа
                mutated_dna[i] = random.choice(self.all_letters)

        return "".join(mutated_dna)

    def train(self, initial_message):
        """Обучает генетический алгоритм на реальных данных из initial_message"""
        self.avg_scores_per_iter = np.zeros(self.NUM_ITER)
        self.best_scores_per_iter = np.zeros(self.NUM_ITER)
        self.best_dna = None
        self.best_mapping = None
        self.best_score = float("-inf")
        dna_scores = {}

        encoded_message = self.encoder.encode(initial_message)

        for i in range(self.NUM_ITER):
            if i > 0:
                self.dna_pool = self.evolve_offspring()

            for dna in self.dna_pool:
                curr_mapping = {
                    original_letter: encoded_letter
                    for original_letter, encoded_letter in zip(self.all_letters, dna)
                }
                curr_decoded_message = self.encoder.decode(encoded_message, curr_mapping)
                dna_scores[dna] = self.m.get_sentence_log_probability(
                    curr_decoded_message
                )

                if dna_scores[dna] > self.best_score:
                    self.best_score = dna_scores[dna]
                    self.best_mapping = curr_mapping
                    self.best_dna = dna

            self.avg_scores_per_iter[i] = np.mean(list(dna_scores.values()))
            self.best_scores_per_iter[i] = self.best_score

            sorted_dna_curr_gen = sorted(
                dna_scores.items(), key=lambda x: x[1], reverse=True
            )

            self.dna_pool = [sequence[0] for sequence in sorted_dna_curr_gen[:5]]

            if i in np.arange(0, 251, 10):
                print(
                    "\n iter: {},".format(i),
                    "log likelihood: {},".format(self.avg_scores_per_iter[i]),
                    "best likelihood so far: {}".format(self.best_score),
                    "\n decoded_message: \n {} \n".format(
                        self.encoder.decode(encoded_message, self.best_mapping)
                    ),
                )
