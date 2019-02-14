import network as nt
import random as rn
import numpy as np

hidden_layer_count_range = range(0, 0)
hidden_layer_neuron_count_range = range(0, 0)
mini_batch_size_range = range(0, 0)
learning_rate_range = np.arange(0.0, 0.0)
regularization_rate_range = np.arange(0.0, 0.0)

training_data = []
epoch_count = 0
evaluation_data = []

class NetworkConfiguration:

    @staticmethod
    def generate():
        hidden_layer_count = rn.randint(hidden_layer_count_range[0], hidden_layer_count_range[-1])
        hidden_layer_layout = np.random.randint(hidden_layer_neuron_count_range[0], hidden_layer_neuron_count_range[-1], hidden_layer_count)
        mini_batch_size = rn.randint(mini_batch_size_range[0], mini_batch_size_range[-1])
        learning_rate = rn.uniform(learning_rate_range[0], learning_rate_range[-1])
        regularization_rate = rn.uniform(regularization_rate_range[0], regularization_rate_range[-1])
        return NetworkConfiguration(hidden_layer_layout, mini_batch_size, learning_rate, regularization_rate)

    def __init__(self, hidden_layer_layout, mini_batch_size, learning_rate, regularization_rate):
        self.hidden_layer_layout = hidden_layer_layout
        self.mini_batch_size = mini_batch_size
        self.learning_rate = learning_rate
        self.regularization_rate = regularization_rate

    def calculate_fitness(self):
        network = nt.Network(np.concatenate(([784], self.hidden_layer_layout, [10])))
        network.train(training_data, epoch_count, self.mini_batch_size, self.learning_rate, self.regularization_rate)
        return network.evaluate_performance(evaluation_data)

    def mutate(self):
        parameter_count = 4.0
        if rn.random() < (1.0 / parameter_count):
            target_hidden_layer_index = rn.randint(0, len(self.hidden_layer_layout) - 1)
            target_hidden_layer_neuron_count = rn.randint(hidden_layer_neuron_count_range[0], hidden_layer_neuron_count_range[-1])
            self.hidden_layer_layout[target_hidden_layer_index] = target_hidden_layer_neuron_count
        elif rn.random() < (1.0 / parameter_count):
            self.mini_batch_size = rn.randint(mini_batch_size_range[0], mini_batch_size_range[-1])
        elif rn.random() < (1.0 / parameter_count):
            self.learning_rate = rn.uniform(learning_rate_range[0], learning_rate_range[-1])
        else:
            self.regularization_rate = rn.uniform(regularization_rate_range[0], regularization_rate_range[-1])

    def recombine(self, group):
        # Randomly select a partner
        partner = rn.choice(group)
        # Recombine with the selected partner
        child_hidden_layer_layout = rn.choice([self.hidden_layer_layout, partner.hidden_layer_layout])
        child_mini_batch_size = rn.choice([self.mini_batch_size, partner.mini_batch_size])
        child_learning_rate = rn.choice([self.learning_rate, partner.learning_rate])
        child_regularization_rate = rn.choice([self.regularization_rate, partner.regularization_rate])
        # Return
        return NetworkConfiguration(child_hidden_layer_layout, child_mini_batch_size, child_learning_rate, child_regularization_rate)