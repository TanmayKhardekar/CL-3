import numpy as np

# Parameters
num_antibodies = 10  # smaller population
mutation_scale = 0.5  # mutation impact, how much an antibody can change during mutation.
num_generations = 20  # number of iterations

# Objective function (calculate_affinity)
# Evaluate the fitness of the population. Lower the value better the likelihood.
def objective_function(x):
    return x**2

# Initialize antibodies
# At first each option is the considered as the solution.
antibodies = np.random.uniform(-10, 10, num_antibodies)

# Run the clonal selection algorithm
for _ in range(num_generations):
    # Evaluate fitness
    fitness = objective_function(antibodies)
    
    # Select the best antibody (lowest fitness)
    best_idx = np.argmin(fitness)
    best_antibody = antibodies[best_idx]
    
    # Generate a mutant from the best antibody
    mutant = best_antibody + np.random.normal(0, mutation_scale)
    
    # Evaluate mutant fitness
    mutant_fitness = objective_function(mutant)
    
    # Replace the worst antibody if mutant is better
    worst_idx = np.argmax(fitness)
    if mutant_fitness < fitness[worst_idx]:
        antibodies[worst_idx] = mutant

# Result
best_idx = np.argmin(objective_function(antibodies))
best_antibody = antibodies[best_idx]
best_fitness = objective_function(best_antibody)

print("Best antibody:", best_antibody)
print("Best fitness:", best_fitness)
