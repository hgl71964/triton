import random
import math

# Function to generate a random binary vector
def generate_random_solution(length):
    return [random.randint(-1, 1) for _ in range(length)]

# Function to calculate the fitness (sum of elements in the vector)
def calculate_fitness(solution):
    return sum(solution)

# Simulated Annealing algorithm
def simulated_annealing(initial_solution, fitness_func, max_iterations, initial_temperature, cooling_rate):
    current_solution = initial_solution[:]
    current_fitness = fitness_func(current_solution)
    best_solution = current_solution[:]
    best_fitness = current_fitness

    temperature = initial_temperature

    for i in range(max_iterations):
        new_solution = current_solution[:]
        index_to_change = random.randint(0, len(new_solution) - 1)
        new_solution[index_to_change] = random.randint(-1, 1)  # Change a random element

        new_fitness = fitness_func(new_solution)
        delta_fitness = new_fitness - current_fitness

        # Metropolis criterion
        if delta_fitness > 0 or random.random() < math.exp(delta_fitness / temperature):
            current_solution = new_solution
            current_fitness = new_fitness

        # Keep track of the best solution found
        if current_fitness > best_fitness:
            best_solution = current_solution[:]
            best_fitness = current_fitness

        # Cooling schedule
        temperature *= cooling_rate

    return best_solution, best_fitness

# Parameters
vector_length = 10
max_iterations = 1000
initial_temperature = 100.0
cooling_rate = 0.95

# Initialize a random solution
initial_solution = generate_random_solution(vector_length)

# Run simulated annealing
best_solution, best_fitness = simulated_annealing(initial_solution, calculate_fitness, max_iterations, initial_temperature, cooling_rate)

# Output results
print("Best Solution:", best_solution)
print("Fitness:", best_fitness)
