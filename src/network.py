import random as rn
import numpy as np

class Network(object):

    def __init__(self, layout):
        self.layer_count = len(layout)
        self.biases = [np.random.randn(neuron_count, 1) for neuron_count in layout[1:]]
        # The weights are initialized as Gaussian random numbers with mean 0 and standard deviation 1,
        # divided by the square root of the number of incoming connections to the neuron. The idea behind
        # the division is to avoid saturating neurons right away, thereby improving the speed at which
        # the network initially learns.
        self.weights = [np.random.randn(rhs_neuron_count, lhs_neuron_count) / np.sqrt(lhs_neuron_count)
            for lhs_neuron_count, rhs_neuron_count in zip(layout[:-1], layout[1:])]

    def train(self, training_data, epoch_count, mini_batch_size, learning_rate, regularization_rate, evaluation_data = None):
        training_example_count = len(training_data)
        if evaluation_data:
            evaluation_example_count = len(evaluation_data)
        for i in xrange(epoch_count):
            rn.shuffle(training_data)
            mini_batches = [training_data[j:j + mini_batch_size] for j in xrange(0, training_example_count, mini_batch_size)]
            for mini_batch in mini_batches:
                self.__process_mini_batch__(mini_batch, learning_rate, regularization_rate, training_example_count)
            if evaluation_data:
                accuracy = float(self.evaluate_performance(evaluation_data)) / float(evaluation_example_count) # Warning can safely be ignored
                print "Accuracy after epoch #{0}: {1}%".format(i + 1, accuracy * 100)

    def __process_mini_batch__(self, mini_batch, learning_rate, regularization_rate, training_example_count):
        eta = learning_rate
        lmd = regularization_rate
        n = training_example_count
        m = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.__process_example__(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # The (1 - eta * (lmd / n)) bit in the expression below is where the L2 regularization (weight decay)
        # step happens. This makes the network prefer smaller weights, which helps against overfitting and
        # prevents certain scenarios where the network can get stuck.
        self.weights = [(1 - eta * (lmd / n)) * w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]

    def __process_example__(self, input, expected_output):
        # Prepare
        a_0 = input
        y = expected_output
        delta_nabla_b = [np.zeros(b.shape) for b in self.biases]
        delta_nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Predict the output
        z_s = []
        a_previous = a_0
        a_s = [a_0]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, a_previous) + b
            z_s.append(z)
            a = __sigmoid__(z)
            a_s.append(a)
            a_previous = a
        # Adjust the weights and biases
        for i in xrange(1, self.layer_count):
            if i == 1: # The last layer since we're iterating over the layers in reverse
                delta = a_s[-i] - y
            else:
                z = z_s[-i]
                delta = np.dot(self.weights[-i + 1].transpose(), delta) * __sigmoid_derivative__(z)
            delta_nabla_b[-i] = delta
            delta_nabla_w[-i] = np.dot(delta, a_s[-i - 1].transpose())
        # Return
        return delta_nabla_b, delta_nabla_w

    def evaluate_performance(self, evaluation_data):
        results = [(np.argmax(self.predict_output(x)), y) for (x, y) in evaluation_data]
        return sum(int(a == y) for (a, y) in results)

    def predict_output(self, input):
        a = input
        for b, w in zip(self.biases, self.weights):
            a = __sigmoid__(np.dot(w, a) + b)
        return a

# Convenience
def __sigmoid__(z):
    return 1.0 / (1.0 + np.exp(-z))

def __sigmoid_derivative__(z):
    return __sigmoid__(z) * (1.0 - __sigmoid__(z))
