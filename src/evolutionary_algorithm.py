import random as rn
import network_configuration as nc

def __calculate_fitness__(network_configuration):
    return network_configuration.calculate_fitness()

def approximate_optimal_network_configuration(population_size, generation_count, parent_fraction, elitist_fraction, mutation_probability):
    # Randomly generate initial population
    population = [nc.NetworkConfiguration.generate() for _ in xrange(population_size)]
    # Iterate
    for i in xrange(generation_count):
        print "Generation {0}".format(i + 1)
        population.sort(key = __calculate_fitness__, reverse = True)
        # Select parents
        parent_count = int(parent_fraction * float(population_size))
        parents = rn.sample(population, parent_count)
        # Select elitists
        elitist_count = int(elitist_fraction * float(population_size))
        elitists = population[:elitist_count]
        # Generate children
        children = []
        child_count = 0
        while True:
            for parent in parents:
                child = parent.recombine(parents)
                children.append(child)
                child_count += 1
                if elitist_count + child_count == population_size:
                    break
            break
        # Randomly mutate children
        for child in children:
            if rn.random() < mutation_probability:
                child.mutate()
        # Update population
        population = elitists + children
    # Return
    return population