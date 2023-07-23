import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from math import factorial
from scipy.special import erfinv, comb
from scipy.stats import uniform, binom, norm
from dataclasses import dataclass
from sklearn.metrics import accuracy_score
import pprint

pp = pprint.PrettyPrinter()

import utils
from utils import (
    estimate_gaussian_params,
    estimate_binomial_params,
    estimate_uniform_params
)


def uniform_generator(a, b, num_samples=100):
    """
    Generates an array of uniformly distributed random numbers within the specified range.

    Parameters:
    - a (float): The lower bound of the range.
    - b (float): The upper bound of the range.
    - num_samples (int): The number of samples to generate (default: 100).

    Returns:
    - array (ndarray): An array of random numbers sampled uniformly from the range [a, b).
    """

    np.random.seed(42)

    ### START CODE HERE ###
    array = np.array(np.random.uniform(a, b, num_samples), dtype=np.dtype(float))
    ### END CODE HERE ###

    return array


def inverse_cdf_gaussian(y, mu, sigma):
    """
    Calculates the inverse cumulative distribution function (CDF) of a Gaussian distribution.

    Parameters:
    - y (float or ndarray): The probability or array of probabilities.
    - mu (float): The mean of the Gaussian distribution.
    - sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
    - x (float or ndarray): The corresponding value(s) from the Gaussian distribution that correspond to the given probability/ies.
    """
    ### START CODE HERE ###
    x = norm.ppf(y, loc=mu, scale=sigma)
    ### END CODE HERE ###

    return x


def gaussian_generator(mu, sigma, num_samples):
    ### START CODE HERE ###

    # Generate an array with num_samples elements that distribute uniformally between 0 and 1
    u = uniform_generator(0, 1, num_samples)

    # Use the uniform-distributed sample to generate Gaussian-distributed data
    # Hint: You need to sample from the inverse of the CDF of the distribution you are generating
    array = inverse_cdf_gaussian(u, mu, sigma)
    ### END CODE HERE ###

    return array


gaussian_0 = gaussian_generator(0, 1, 1000)
gaussian_1 = gaussian_generator(5, 3, 1000)
gaussian_2 = gaussian_generator(10, 5, 1000)


# utils.plot_gaussian_distributions(gaussian_0, gaussian_1, gaussian_2)

def inverse_cdf_binomial(y, n, p):
    """
    Calculates the inverse cumulative distribution function (CDF) of a binomial distribution.

    Parameters:
    - y (float or ndarray): The probability or array of probabilities.
    - n (int): The number of trials in the binomial distribution.
    - p (float): The probability of success in each trial.

    Returns:
    - x (float or ndarray): The corresponding value(s) from the binomial distribution that correspond to the given probability/ies.
    """

    ### START CODE HERE ###
    x = binom.ppf(y, n, p)
    ### END CODE HERE ###

    return x


def binomial_generator(n, p, num_samples):
    """
    Generates an array of binomially distributed random numbers.

    Args:
        n (int): The number of trials in the binomial distribution.
        p (float): The probability of success in each trial.
        num_samples (int): The number of samples to generate.

    Returns:
        array: An array of binomially distributed random numbers.
    """
    ### START CODE HERE ###

    # Generate an array with num_samples elements that distribute uniformally between 0 and 1
    u = uniform_generator(0, 1, num_samples)

    # Use the uniform-distributed sample to generate binomial-distributed data
    # Hint: You need to sample from the inverse of the CDF of the distribution you are generating
    array = inverse_cdf_binomial(u, n, p)
    ### END CODE HERE ###

    return array


binomial_0 = binomial_generator(12, 0.4, 1000)
binomial_1 = binomial_generator(15, 0.5, 1000)
binomial_2 = binomial_generator(25, 0.8, 1000)

# utils.plot_binomial_distributions(binomial_0, binomial_1, binomial_2)


FEATURES = ["height", "weight", "bark_days", "ear_head_ratio"]


@dataclass
class params_gaussian:
    mu: float
    sigma: float

    def __repr__(self):
        return f"params_gaussian(mu={self.mu:.3f}, sigma={self.sigma:.3f})"


@dataclass
class params_binomial:
    n: int
    p: float

    def __repr__(self):
        return f"params_binomial(n={self.n:.3f}, p={self.p:.3f})"


@dataclass
class params_uniform:
    a: int
    b: int

    def __repr__(self):
        return f"params_uniform(a={self.a:.3f}, b={self.b:.3f})"


breed_params = {
    0: {
        "height": params_gaussian(mu=35, sigma=1.5),
        "weight": params_gaussian(mu=20, sigma=1),
        "bark_days": params_binomial(n=30, p=0.8),
        "ear_head_ratio": params_uniform(a=0.6, b=0.1)
    },

    1: {
        "height": params_gaussian(mu=30, sigma=2),
        "weight": params_gaussian(mu=25, sigma=5),
        "bark_days": params_binomial(n=30, p=0.5),
        "ear_head_ratio": params_uniform(a=0.2, b=0.5)
    },

    2: {
        "height": params_gaussian(mu=40, sigma=3.5),
        "weight": params_gaussian(mu=32, sigma=3),
        "bark_days": params_binomial(n=30, p=0.3),
        "ear_head_ratio": params_uniform(a=0.1, b=0.3)
    }

}


def generate_data_for_breed(breed, features, n_samples, params):
    """
    Generate synthetic data for a specific breed of dogs based on given features and parameters.

    Parameters:
        - breed (str): The breed of the dog for which data is generated.
        - features (list[str]): List of features to generate data for (e.g., "height", "weight", "bark_days", "ear_head_ratio").
        - n_samples (int): Number of samples to generate for each feature.
        - params (dict): Dictionary containing parameters for each breed and its features.

    Returns:
        - df (pandas.DataFrame): A DataFrame containing the generated synthetic data.
            The DataFrame will have columns for each feature and an additional column for the breed.
    """

    df = pd.DataFrame()

    for feature in features:
        match feature:
            case "height" | "weight":
                df[feature] = gaussian_generator(params[breed][feature].mu, params[breed][feature].sigma, n_samples)

            case "bark_days":
                df[feature] = binomial_generator(params[breed][feature].n, params[breed][feature].p, n_samples)

            case "ear_head_ratio":
                df[feature] = uniform_generator(params[breed][feature].a, params[breed][feature].b, n_samples)

    df["breed"] = breed

    return df


# Generate data for each breed
df_0 = generate_data_for_breed(breed=0, features=FEATURES, n_samples=1200, params=breed_params)
df_1 = generate_data_for_breed(breed=1, features=FEATURES, n_samples=1350, params=breed_params)
df_2 = generate_data_for_breed(breed=2, features=FEATURES, n_samples=900, params=breed_params)

# Concatenate all breeds into a single dataframe
df_all_breeds = pd.concat([df_0, df_1, df_2]).reset_index(drop=True)

# Shuffle the data
df_all_breeds = df_all_breeds.sample(frac=1)

# Print the dataframe
df_all_breeds.head(10)

# Define a 70/30 training/testing split
split = int(len(df_all_breeds) * 0.7)

# Do the split
df_train = df_all_breeds[:split].reset_index(drop=True)
df_test = df_all_breeds[split:].reset_index(drop=True)


def pdf_uniform(x, a, b):
    """
    Calculates the probability density function (PDF) for a uniform distribution between 'a' and 'b' at a given point 'x'.

    Args:
        x (float): The value at which the PDF is evaluated.
        a (float): The lower bound of the uniform distribution.
        b (float): The upper bound of the uniform distribution.

    Returns:
        float: The PDF value at the given point 'x'. Returns 0 if 'x' is outside the range [a, b].
    """
    ### START CODE HERE ###
    if x < a or x > b:
        pdf = 0
    else:
        pdf = 1 / (b - a)
    ### END CODE HERE ###

    return pdf


def pdf_gaussian(x, mu, sigma):
    """
    Calculate the probability density function (PDF) of a Gaussian distribution at a given value.

    Args:
        x (float or array-like): The value(s) at which to evaluate the PDF.
        mu (float): The mean of the Gaussian distribution.
        sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
        float or ndarray: The PDF value(s) at the given point(s) x.
    """

    ### START CODE HERE ###
    pdf = norm.pdf(x, loc=mu, scale=sigma)

    ### END CODE HERE ###

    return pdf


def pdf_binomial(x, n, p):
    """
    Calculate the probability mass function (PMF) of a binomial distribution at a specific value.

    Args:
        x (int): The value at which to evaluate the PMF.
        n (int): The number of trials in the binomial distribution.
        p (float): The probability of success for each trial.

    Returns:
        float: The probability mass function (PMF) of the binomial distribution at the specified value.
    """

    ### START CODE HERE ###
    pdf = binom.pmf(x, n, p)
    ### END CODE HERE ###

    return pdf


def compute_training_params(df, features):
    """
    Computes the estimated parameters for training a model based on the provided dataframe and features.

    Args:
        df (pandas.DataFrame): The dataframe containing the training data.
        features (list): A list of feature names to consider.

    Returns:
        tuple: A tuple containing two dictionaries:
            - params_dict (dict): A dictionary that contains the estimated parameters for each breed and feature.
            - probs_dict (dict): A dictionary that contains the proportion of data belonging to each breed.
    """

    # Dict that should contain the estimated parameters
    feature_estimate = None
    params_dict = {}

    # Dict that should contain the proportion of data belonging to each class
    probs_dict = {}

    # print(df)
    for index, row in df.iterrows():
        current_breed_val = row.get('breed')

        # Slice the original df to only include data for the current breed and the feature columns
        # For reference in slicing with pandas, you can use the df_breed.groupby function followed by .get_group
        # or you can use the syntax df[df['breed'] == group]
        df_breed = df[df["breed"] == current_breed_val][features]
        # Save the probability of each class (breed) in the probabilities dict
        # You can find the number of rows in a dataframe by using len(dataframe)
        if current_breed_val not in probs_dict:
            probs_dict[current_breed_val] = round(len(df_breed) / len(df), 3)
        # Initialize the inner dict
        inner_dict = {}  # @KEEP
        # Loop over the columns of the sliced dataframe
        # You can get the columns of a dataframe like this: dataframe.columns
        for j in df_breed:
            feat_found = df_breed.get(j)
            match j:
                case "height" | "weight":
                    # Estimate parameters depending on the distribution of the current feature
                    # and save them in the corresponding dataclass object
                    m, s = estimate_gaussian_params(feat_found)
                    feature_estimate = params_gaussian(m, s)
                case "bark_days":
                    # Estimate parameters depending on the distribution of the current feature
                    # and save them in the corresponding dataclass object
                    n, p = estimate_binomial_params(feat_found)
                    feature_estimate = params_binomial(n, p)
                case "ear_head_ratio":
                    # Estimate parameters depending on the distribution of the current feature
                    # and save them in the corresponding dataclass object
                    a, b = estimate_uniform_params(feat_found)
                    feature_estimate = params_uniform(a, b)
            # Save the dataclass object within the inner dict
            inner_dict[j] = feature_estimate

        # Save inner dict within outer dict
        params_dict[current_breed_val] = inner_dict
    ### END CODE HERE ###

    return params_dict, probs_dict


train_params, train_class_probs = compute_training_params(df_train, FEATURES)


def prob_of_X_given_C(X, features, breed, params_dict):
    """
    Calculate the conditional probability of X given a specific breed, using the given features and parameters.

    Args:
        X (list): List of feature values for which the probability needs to be calculated.
        features (list): List of feature names corresponding to the feature values in X.
        breed (str): The breed for which the probability is calculated.
        params_dict (dict): Dictionary containing the parameters for different breeds and features.

    Returns:
        float: The conditional probability of X given the specified breed.
    """

    if len(X) != len(features):
        print("X and list of features should have the same length")
        return 0

    probability = 1.0

    ### START CODE HERE ###
    probability_f = None
    for feat, val in zip(features, X):
        # Get the relevant parameters from params_dict
        params = params_dict[breed][feat]
        match feat:
            # You can add add as many case statements as you see fit
            case "height" | "weight":
                # Compute the relevant pdf given the distribution and the estimated parameters
                probability_f = pdf_gaussian(val, params.mu, params.sigma)

            case "bark_days":
                # Compute the relevant pdf given the distribution and the estimated parameters
                probability_f = pdf_binomial(val, params.n, params.p)

            case "ear_head_ratio":
                # Compute the relevant pdf given the distribution and the estimated parameters
                probability_f = pdf_uniform(val, params.a, params.b)

        # Multiply by probability of current feature
        probability *= probability_f

    ### END CODE HERE ###

    return probability


example_dog = df_test[FEATURES].loc[0]
example_breed = df_test[["breed"]].loc[0]["breed"]


# print(f"Example dog has breed {example_breed} and features: height = {example_dog['height']:.2f}, weight = {example_dog['weight']:.2f}, bark_days = {example_dog['bark_days']:.2f}, ear_head_ratio = {example_dog['ear_head_ratio']:.2f}\n")
# print(
#     f"Probability of these features if dog is classified as breed 0: {prob_of_X_given_C([*example_dog], FEATURES, 0, train_params)}")


# print(f"Probability of these features if dog is classified as breed 1: {prob_of_X_given_C([*example_dog], FEATURES, 1, train_params)}")
# print(f"Probability of these features if dog is classified as breed 2: {prob_of_X_given_C([*example_dog], FEATURES, 2, train_params)}")
#

def predict_breed(X, features, params_dict, probs_dict):
    """
    Predicts the breed based on the input and features.

    Args:
        X (array-like): The input data for prediction.
        features (array-like): The features used for prediction.
        params_dict (dict): A dictionary containing parameters for different breeds.
        probs_dict (dict): A dictionary containing probabilities for different breeds.

    Returns:
        int: The predicted breed index.
    """

    ### START CODE HERE ###
    posterior_breed_0 = prob_of_X_given_C(X, features, 0, params_dict) * probs_dict[0]
    posterior_breed_1 = prob_of_X_given_C(X, features, 1, params_dict) * probs_dict[1]
    posterior_breed_2 = prob_of_X_given_C(X, features, 2, params_dict) * probs_dict[2]

    # Save the breed with the maximum posterior
    # Hint: You can create a numpy array with the posteriors and then use np.argmax
    values = [posterior_breed_0, posterior_breed_1, posterior_breed_2]
    array = np.array(values)
    prediction = np.argmax(array)
    ### END CODE HERE ###

    return prediction


example_pred = predict_breed([*example_dog], FEATURES, train_params, train_class_probs)
# print(f"Example dog has breed {example_breed} and Naive Bayes classified it as {example_pred}")


# Load the dataset
emails = pd.read_csv('emails.csv')


# Helper function that converts text to lowercase and splits words into a list
def process_email(text):
    """
    Processes the given email text by converting it to lowercase, splitting it into words,
    and returning a list of unique words.

    Parameters:
    - text (str): The email text to be processed.

    Returns:
    - list: A list of unique words extracted from the email text.
    """

    text = text.lower()
    return list(set(text.split()))


# Create an extra column with the text converted to a lower-cased list of words
emails['words'] = emails['text'].apply(process_email)

# Show the first 5 rows
emails.head(5)


def word_freq_per_class(df):
    """
    Calculates the frequency of words in each class (spam and ham) based on a given dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe containing email data,
        with a column named 'words' representing the words in each email.

    Returns:
        dict: A dictionary containing the frequency of words in each class.
        The keys of the dictionary are words, and the values are nested dictionaries with keys
        'spam' and 'ham' representing the frequency of the word in spam and ham emails, respectively.
    """

    word_freq_dict = {}

    ### START CODE HERE ###

    # Hint: You can use the iterrows() method to iterate over the rows of a dataframe.
    # This method yields an index and the data in the row so you can ignore the first returned value.
    for _, row in df.iterrows():
        # Iterate over the words in each email
        for word in row.words:
            # Check if word doesn't exist within the dictionary
            if word not in word_freq_dict:
                # If word doesn't exist, initialize the count at 0
                word_freq_dict[word] = {'spam': 0, 'ham': 0}

            # Check if the email was spam
            match row['spam']:
                case 0:
                    # If ham then add 1 to the count of ham
                    word_freq_dict[word]['ham'] += 1
                case 1:
                    # If spam then add 1 to the count of spam
                    word_freq_dict[word]['spam'] += 1

    ### END CODE HERE ###

    return word_freq_dict


word_freq = word_freq_per_class(emails)


# print(f"Frequency in both classes for word 'lottery': {word_freq['lottery']}\n")
# print(f"Frequency in both classes for word 'sale': {word_freq['sale']}\n")

# try:
#     word_freq['asdfg']
# except KeyError:
#     print("Word 'asdfg' not in corpus")


def class_frequencies(df):
    """
    Calculate the frequencies of classes in a DataFrame.

    Args:
        df (DataFrame): The input DataFrame containing a column 'spam' indicating class labels.

    Returns:
        dict: A dictionary containing the frequencies of the classes.
            The keys are 'spam' and 'ham', representing the class labels.
            The values are the corresponding frequencies in the DataFrame.
    """

    ### START CODE HERE ###
    spam_count = 0
    ham_count = 0
    for index, row in df.iterrows():
        indicator = row.get('spam')
        if indicator == 1:
            spam_count += 1
        else:
            ham_count += 1

    class_freq_dict = {
        "spam": spam_count,
        "ham": ham_count
    }

    ### END CODE HERE ###

    return class_freq_dict


class_freq = class_frequencies(emails[:10])
# print(f"Small dataset:\n\nThe frequencies for each class are {class_freq}\n")
# print(f"The proportion of spam in the dataset is: {100 * class_freq['spam'] / len(emails[:10]):.2f}%\n")
# print(f"The proportion of ham in the dataset is: {100 * class_freq['ham'] / len(emails[:10]):.2f}%\n")

class_freq = class_frequencies(emails)


# print(f"\nFull dataset:\n\nThe frequencies for each class are {class_freq}\n")
# print(f"The proportion of spam in the dataset is: {100 * class_freq['spam'] / len(emails):.2f}%\n")
# print(f"The proportion of ham in the dataset is: {100 * class_freq['ham'] / len(emails):.2f}%")


def naive_bayes_classifier(text, word_freq=word_freq, class_freq=class_freq):
    """
    Implements a naive Bayes classifier to determine the probability of an email being spam.

    Args:
        text (str): The input email text to classify.

        word_freq (dict): A dictionary containing word frequencies in the training corpus.
        The keys are words, and the values are dictionaries containing frequencies for 'spam' and 'ham' classes.

        class_freq (dict): A dictionary containing class frequencies in the training corpus.
        The keys are class labels ('spam' and 'ham'), and the values are the respective frequencies.

    Returns:
        float: The probability of the email being spam.

    """

    text = text.lower()
    words = set(text.split())
    cumulative_product_spam = 1.0
    cumulative_product_ham = 1.0
    prob_of_unknown_word = class_freq['spam'] / (class_freq['spam'] + class_freq['ham'])

    # Iterate over the words in the email
    for word in words:
        # You should only include words that exist in the corpus in your calculations
        if word in word_freq:
            cumulative_product_spam *= word_freq[word]['spam']
            cumulative_product_ham *= word_freq[word]['ham']
        else:
            cumulative_product_spam *= prob_of_unknown_word
            cumulative_product_ham *= 1.0 - prob_of_unknown_word

    # Calculate the likelihood of the words appearing in the email given that it is spam
    likelihood_word_given_spam = cumulative_product_spam / class_freq['spam']

    # Calculate the likelihood of the words appearing in the email given that it is ham
    likelihood_word_given_ham = cumulative_product_ham / class_freq['ham']

    # Calculate the posterior probability of the email being spam given that the words appear in the email (the probability of being a spam given the email content)
    prob_spam = likelihood_word_given_spam / (likelihood_word_given_spam + likelihood_word_given_ham)

    return prob_spam


msg = "enter the lottery to win three million dollars"
print(f"Probability of spam for email '{msg}': {100 * naive_bayes_classifier(msg):.2f}%\n")

msg = "meet me at the lobby of the hotel at nine am"
print(f"Probability of spam for email '{msg}': {100*naive_bayes_classifier(msg):.2f}%\n")

msg = "9898 asjfkjfdj"
print(f"Probability of spam for email '{msg}': {100*naive_bayes_classifier(msg):.2f}%\n")
