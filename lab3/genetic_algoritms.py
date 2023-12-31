import random
import matplotlib.pyplot as plt

ONE_MAX_LENGTH = 100

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 50

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


class FitnessMax():
    def __init__(self):
        self.values = [0]


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()


def oneMaxFitness(individual):
    return sum(individual),


def individualCreator():
    return Individual([random.randint(0, 1) for i in range(ONE_MAX_LENGTH)])


def populationCreator(n=0):
    return list([individualCreator() for i in range(n)])


population = populationCreator(n=POPULATION_SIZE)
generatorCounter = 0

fitnessValues = list(map(oneMaxFitness, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue


maxFitnessValues = []
meanFitnessValues = []


def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind


def selToTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len-1), random.randint(0, p_len-1), random.randint(0, p_len-1)

        offspring.append(max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring


def cxOnePoint(child1, child2):
    s = random.randint(2, len(child1)-3)
    child1[s:], child2[s:] = child2[s:], child1[s:]


def mutFlipBit(mutant, indpb=0.01):
    for index in range(len(mutant)):
        if random.random() < indpb:
            mutant[index] = 0 if mutant[index] == 1 else 1


fitnessValues = [individual.fitness.values[0] for individual in population]


while max(fitnessValues) < ONE_MAX_LENGTH and generatorCounter < MAX_GENERATIONS:
    generatorCounter += 1
    offspring = selToTournament(population, len(population))
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            cxOnePoint(child1, child2)

    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutFlipBit(mutant, indpb=1.0/ONE_MAX_LENGTH)

    freshFitnessValues = list(map(oneMaxFitness, offspring))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    population[:] = offspring

    fitnessValues = [ind.fitness.values[0] for ind in population]

    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues)/len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)

    print(f"Поколение {generatorCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ. = {meanFitness}")

    best_index = fitnessValues.index(max(fitnessValues))
    print("Лучший индивидум = ", *population[best_index], "\n")

plt.plot(maxFitnessValues, color="red")
plt.plot(meanFitnessValues, color="blue")
plt.xlabel('Поколение')
plt.ylabel("Макс/средняя приспособленность")
plt.title('Зависимость максимальной и средней приспособленности от поколеия')
plt.show()