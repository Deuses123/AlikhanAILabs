import random
import re


class Encoder:
    def __init__(self):
        self.__build_cipher_mapping()

    def __build_cipher_mapping(self):
        """Создание шифра для замены букв"""
        random.seed(7)
        letters_original = list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        letters_shuffled = random.sample(
            list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя'), len(list('абвгдеёжзийклмнопрстуфхцчшщъыьэюя'))
        )
        self.encoder_cipher_mapping = {
            letters_original[i]: letters_shuffled[i] for i in range(len(letters_original))
        }

    def encode(self, message):
        """Метод для кодирования сообщения с использованием шифра"""
        message = re.sub("[^а-яёА-ЯЁ]", " ", message)
        message_tokens = list(message.lower())
        for i in range(len(message)):
            if message_tokens[i] in self.encoder_cipher_mapping:
                message_tokens[i] = self.encoder_cipher_mapping[message_tokens[i]]
        encoded_message = "".join(message_tokens)
        return encoded_message

    def decode(self, encoded_message, mapping=None):
        """Метод для декодирования сообщения с использованием отображения шифра"""
        if mapping is None:
            mapping = self.encoder_cipher_mapping
        message_tokens = list(encoded_message.lower())
        for i in range(len(encoded_message)):
            if message_tokens[i] in mapping:
                message_tokens[i] = mapping[message_tokens[i]]
        decoded_message = "".join(message_tokens)
        return decoded_message