import numpy as np


def expected_value(numbers):
    return np.mean(numbers)


def sum_of_expected_values(numbers):
    return sum(numbers)


def calculate_variance(arr):
    n = len(arr)
    if n < 2:
        return None  # Variance is undefined for a single value

    mean = sum(arr) / n
    variance = sum((x - mean) ** 2 for x in arr) / (n - 1)
    return variance


def joint_distribution(prob1, prob2):
    if len(prob1) != len(prob2):
        raise ValueError("Both probability lists should have the same length.")

    joint_prob = [p1 * p2 for p1, p2 in zip(prob1, prob2)]

    return joint_prob


print(calculate_variance([-1,0,2]))