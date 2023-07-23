import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils
import maths.assignments.probability_and_statistics.probability as prob_funcs


def roll_fair_dice(dice_sides: int, num_rolls: int):
    rng = np.random.default_rng()
    r_ints = rng.integers(low=1, high=dice_sides, size=num_rolls)
    return np.array(r_ints)


# Exercise 1
values = [0.167, 0.167, 0.167, 0.167, 0.167, 0.167]
variance_1 = prob_funcs.calculate_variance(values)
mean_1 = prob_funcs.expected_value(values)
# print('# Exercise 1')
# print("variance {0}".format(variance_1))
# print("mean {0}".format(mean_1))
# print('\n')

# Exercise 2
# print('# Exercise 2')
dice_roll_result = roll_fair_dice(6, 2)
dice_sum = prob_funcs.sum_of_expected_values(dice_roll_result)
# print(dice_sum)
# print('\n')

# Exercise 3
# print('# Exercise 3')
dice_roll_result = roll_fair_dice(4, 2)
dice_sum = prob_funcs.sum_of_expected_values(dice_roll_result)


# print(dice_sum)


def dice_sum_probabilities(sides=4):
    # Initialize a dictionary to store the sums and their probabilities
    sum_dict = {i: 0 for i in range(2, sides * 2 + 1)}

    # Loop through all possible outcomes of the two dice
    for i in range(1, sides + 1):
        for j in range(1, sides + 1):
            # Increment the count for the sum of the dice
            sum_dict[i + j] += 1

    # Calculate the probabilities by dividing by the total number of outcomes
    for key in sum_dict:
        sum_dict[key] /= (sides ** 2)

    return sum_dict


# print(dice_sum_probabilities())
# print('\n')


# Exercise 4
# print('# Exercise 4')


def dice_statistics(sides=4):
    # Generate all possible sums
    sums = [i + j for i in range(1, sides + 1) for j in range(1, sides + 1)]

    # Calculate mean and variance
    mean = np.mean(sums)
    variance = np.var(sums)

    # Covariance is 0 as throws are independent
    covariance = 0

    return mean, variance, covariance


# print(dice_statistics())
# print('\n')

# Exercise 5
# print('# Exercise 5')


def loaded_dice_histogram(sides=4):
    # Probabilities for each side of the dice
    probs = [1 / 6, 1 / 3, 1 / 6, 1 / 6]

    # Initialize a dictionary to store the sums and their probabilities
    sum_dict = {i: 0 for i in range(2, sides * 2 + 1)}

    # Loop through all possible outcomes of the two dice
    for i in range(1, sides + 1):
        for j in range(1, sides + 1):
            # Add the probability for this combination to the sum's probability
            sum_dict[i + j] += probs[i - 1] * probs[j - 1]

    # Generate lists of the sums and their probabilities for plotting
    sums = list(sum_dict.keys())
    probabilities = list(sum_dict.values())

    # Create the histogram
    plt.bar(sums, probabilities)
    plt.xlabel('Sum')
    plt.ylabel('Probability')
    plt.title('Histogram of Sums for Loaded Dice')
    plt.show()


# loaded_dice_histogram()


# print('# Exercise 6')


def loaded_dice_cumulative(sides=6):
    # Probabilities for each side of the dice
    probs = [1 / 7, 1 / 7, 2 / 7, 1 / 7, 1 / 7, 1 / 7]

    # Initialize a dictionary to store the sums and their probabilities
    sum_dict = {i: 0 for i in range(2, sides * 2 + 1)}

    # Loop through all possible outcomes of the two dice
    for i in range(1, sides + 1):
        for j in range(1, sides + 1):
            # Add the probability for this combination to the sum's probability
            sum_dict[i + j] += probs[i - 1] * probs[j - 1]

    # Generate a sorted list of the sums
    sums = sorted(list(sum_dict.keys()))

    # Initialize cumulative probability
    cumulative_prob = 0

    for s in sums:
        # Add the probability of this sum to the cumulative probability
        cumulative_prob += sum_dict[s]

        # Check if the cumulative probability is greater than 0.5
        if cumulative_prob > 0.5:
            return s - 1  # Return the previous sum

    return None  # No sum found that yields a cumulative probability less than 0.5


# print('# Exercise 7')


def dice_game_histogram(sides=6):
    probs = [1 / sides for _ in range(sides)]

    # Initialize a dictionary to store the results and their probabilities
    result_dict = {i: 0 for i in range(1, sides * 2)}

    # First, calculate probabilities if we throw only once
    for i in range(4, sides + 1):
        result_dict[i] += probs[i - 1]

    # Then, calculate probabilities if we throw twice
    for i in range(1, 4):
        for j in range(1, sides + 1):
            result_dict[i + j] += probs[i - 1] * probs[j - 1]

    # Create the histogram of PMF
    plt.bar(result_dict.keys(), result_dict.values())
    plt.xlabel('Sum of Rolls')
    plt.ylabel('Probability')
    plt.title('Probability Mass Function for Dice Game')
    plt.show()


# dice_game_histogram()

print('# Exercise 8')

def dice_game_histogram_2(sides=6):
    # Probabilities for each side of the dice
    probs = [1/sides for _ in range(sides)]

    # Initialize a dictionary to store the results and their probabilities
    result_dict = {i: 0 for i in range(1, sides*2 + 1)}

    # First, calculate probabilities if we throw only once
    for i in range(1, 3):
        result_dict[i] += probs[i-1]

    # Then, calculate probabilities if we throw twice
    for i in range(3, sides + 1):
        for j in range(1, sides + 1):
            result_dict[i + j] += probs[i-1] * probs[j-1]

    # Create the histogram of PMF
    plt.bar(result_dict.keys(), result_dict.values())
    plt.xlabel('Sum of Rolls')
    plt.ylabel('Probability')
    plt.title('Probability Mass Function for Dice Game')
    plt.show()

dice_game_histogram_2()
