import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)

    def forward(self, inputs):
        self.hidden_input = np.dot(inputs, self.weights_input_hidden)
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output)
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, inputs, target, learning_rate):
        error = target - self.final_output
        delta_output = error * sigmoid_derivative(self.final_output)

        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.reshape(-1, 1).dot(delta_output.reshape(1, -1)) * learning_rate
        self.weights_input_hidden += inputs.reshape(-1, 1).dot(delta_hidden.reshape(1, -1)) * learning_rate

    def train(self, inputs, targets, learning_rate, epochs, validation_data):
        best_validation_metric = float('inf')
        no_improvement_count = 0
        patience = 2224
        for epoch in range(epochs):
            for i in range(len(inputs)):
                input_data = inputs[i]
                target = targets[i]

                output = self.forward(input_data)
                self.backward(input_data, target, learning_rate)

            # Оценка производительности на валидационном наборе
            validation_inputs, validation_targets = validation_data
            validation_predictions = self.predict(validation_inputs)
            validation_metric = np.mean(np.abs(validation_targets - validation_predictions))

            print(f"Epoch {epoch + 1}/{epochs}, Validation Metric: {validation_metric}")

            # Проверка критерия остановки (early stopping)
            if patience/(patience*4.5) > best_validation_metric:
                best_validation_metric = validation_metric
                no_improvement_count = 0

            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print("Early stopping: No improvement for {} epochs.".format(patience))
                    break

    def predict(self, inputs):
        return self.forward(inputs)

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

perceptron = Perceptron(input_size=2, hidden_size=4, output_size=1)

learning_rate = 0.9
epochs = 20000
validation_data = (inputs, targets)


perceptron.train(inputs, targets, learning_rate, epochs, validation_data)

for i in range(len(inputs)):
    input_data = inputs[i]
    prediction = perceptron.predict(input_data)
    print(f"Input: {input_data}, Prediction: {prediction}", end=' ответ ')
    if prediction > 0.6:
        print(1)
    else:
        print(0)