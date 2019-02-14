import numpy as np
import mnist_data_loader
import network_configuration as nc
import evolutionary_algorithm
import network as nt
import operator as op

# Load the MNIST data
full_training_data, full_validation_data, full_test_data = mnist_data_loader.load_data()

# Set the network configuration generation & mutation parameters
nc.hidden_layer_count_range = range(1, 4)
nc.hidden_layer_neuron_count_range = range(24, 128)
nc.mini_batch_size_range = range(4, 24)
nc.learning_rate_range = np.arange(0.2, 2.0)
nc.regularization_rate_range = np.arange(0.2, 4.0)

# Set the network configuration evaluation parameters
nc.training_data = full_training_data[:1000]
nc.epoch_count = 12
nc.evaluation_data = full_validation_data[:1000]

# Search for good network configurations using a genetic algorithm
network_configurations = evolutionary_algorithm.approximate_optimal_network_configuration(population_size = 80, generation_count = 8,
    parent_fraction = 0.24, elitist_fraction = 0.16, mutation_probability = 0.06)[:8]

print "--------"

# Log the best network configurations
for nc in network_configurations:
    print "Layout: {0}, Mini Batch Size: {1}, Learning Rate: {2}, Regularization Rate: {3}".format(
        np.concatenate(([784], nc.hidden_layer_layout, [10])), nc.mini_batch_size, nc.learning_rate, nc.regularization_rate)

print "--------"

# Train networks using the best network configurations
epoch_count = 24
networks = [nt.Network(np.concatenate(([784], nc.hidden_layer_layout, [10]))) for nc in network_configurations]
for i in xrange(len(network_configurations)):
    network = networks[i]
    nc = network_configurations[i]
    print "Training network {0}...".format(i + 1)
    network.train(full_training_data, epoch_count, nc.mini_batch_size, nc.learning_rate, nc.regularization_rate, full_validation_data)

print "--------"

# Evaluate the test data as a group
score = 0
for x, y in full_test_data:
    individual_predictions = [nt.predict_output(x) for nt in networks]
    group_prediction = np.argmax(reduce(op.add, individual_predictions))
    if group_prediction == y:
        score += 1

print "Accuracy: {0}%".format((float(score) / float(len(full_test_data))) * 100)