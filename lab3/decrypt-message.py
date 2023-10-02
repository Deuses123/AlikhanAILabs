import random
import string
from cryptography.fernet import Fernet

# Генерация случайного числового ключа для эволюционного алгоритма
def generate_random_key():
    return random.randint(1, 25)  # Генерируем случайное число от 1 до 25 (включительно)

# Функция шифрования сообщения с использованием числового ключа
def encrypt_message(message, key):
    encrypted_message = ''
    for char in message:
        if char.isalpha():
            shifted_char = chr(((ord(char) - ord('A') + key) % 26) + ord('A'))
            encrypted_message += shifted_char
        else:
            encrypted_message += char
    return encrypted_message

# Функция дешифрования сообщения с использованием числового ключа
def decrypt_message(message, key):
    return encrypt_message(message, -key)

# Оценка качества ключа с помощью функции приспособленности
def calculate_fitness(key, target_message):
    decoded_message = decrypt_message(target_message, key)
    fitness = 0
    for i in range(len(target_message)):
        if target_message[i] == decoded_message[i]:
            fitness += 1
    return fitness

# Оценка ключа на допустимость
def is_valid_key(key):
    return 1 <= key <= 25  # Проверяем, что ключ находится в допустимом диапазоне

# Параметры эволюционного алгоритма
population_size = 100
mutation_rate = 1
generations = 100

# Целевое сообщение для кодирования и декодирования
target_message = "Hello, World!"

# Генерируем начальную популяцию числовых ключей
population = [generate_random_key() for _ in range(population_size)]

# Основной цикл эволюции
for generation in range(generations):
    fitness_scores = [calculate_fitness(key, target_message) for key in population]

    # Выбираем лучшие ключи
    best_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    best_keys = [population[i] for i in best_indices[:10]]  # Выбираем 10 лучших ключей

    new_population = []

    while len(new_population) < population_size:
        parent1, parent2 = random.choices(best_keys, k=2)
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        new_population.append(child)

    population = new_population

    # Выводим лучший ключ текущего поколения
    best_key = best_keys[0]
    best_fitness = fitness_scores[best_indices[0]]
    print(f"Generation {generation}: Best Key = {best_key}, Fitness = {best_fitness}")

# Выводим окончательный результат
best_key = best_keys[0]
best_fitness = fitness_scores[best_indices[0]]
print(f"Final Best Key = {best_key}, Fitness = {best_fitness}")
