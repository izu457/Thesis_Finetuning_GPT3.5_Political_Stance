# load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

voted_docs_final_EP9 = pd.read_csv('data/voted_laws_final_EP9.csv')
voted_docs_final_EP9.columns


def get_random_guess(data):
    # create new dataframe with same index/ Vote IDs
    data = pd.DataFrame(index=data.index)
    # add random guesses between 0 and 1 rounded to two decimals for each group
    data['ECR%'] = [round(random.random(), 2) for _ in range(len(data))]
    data['EPP%'] = [round(random.random(), 2) for _ in range(len(data))]
    data['Greens/EFA%'] = [round(random.random(), 2) for _ in range(len(data))]
    data['EFD/IDG%'] = [round(random.random(), 2) for _ in range(len(data))]
    data['NI%'] = [round(random.random(), 2) for _ in range(len(data))]
    data['REG%'] = [round(random.random(), 2) for _ in range(len(data))]
    data['S&D%'] = [round(random.random(), 2) for _ in range(len(data))]
    data['The Left%'] = [round(random.random(), 2) for _ in range(len(data))]
    # guess 0 or 1 for majorities
    data['General Majority'] = [random.choice([0, 1]) for _ in range(len(data))]
    data['Right Majority'] = [random.choice([0, 1]) for _ in range(len(data))]
    data['Left Majority'] = [random.choice([0, 1]) for _ in range(len(data))]
    data['Consensus'] = [random.choice([0, 1]) for _ in range(len(data))]
    return data

# test function
random_guess_EP9 = pd.DataFrame(index=voted_docs_final_EP9["Vote ID"])
random_guess_EP9 = get_random_guess(random_guess_EP9)
print(random_guess_EP9.shape)

# function to calculate accuracy
def get_accuracy_df (original_df, test_df, tolerance_interval):
    """"
    Function to calculate the accuracy of a test dataframe compared to an original dataframe.
    Takes as arguments the original dataframe and the test dataframe and a tolerance interval.
    For each index in the original data frame, it finds the corresponding index in the test data frame
    and checks if the values in the test data frame are within the tolerance interval of the original 
    data frame.
    Calculates accuracy as the number of correct guesses divided by the number of guesses.
    Returns the accuracy.
    """
    # create a new dataframe with the same index as the original dataframe
    accuracy_df = pd.DataFrame(index=test_df.index)
    # add a column for each column in the original dataframe
    for column in test_df.columns:
        accuracy_df[column] = 0
    # loop through the rows of the test dataframe
    for index in test_df.index:
        # loop through the columns of the test dataframe
        for column in test_df.columns:
            # check if the value in the test dataframe is within the tolerance interval of the original dataframe
            if (original_df.loc[index, column] - tolerance_interval) <= test_df.loc[index, column] <= (original_df.loc[index, column] + tolerance_interval):
                # if it is, add 1 to the corresponding cell in the accuracy dataframe
                accuracy_df.loc[index, column] = 1
    # calculate the accuracy as the sum of the accuracy dataframe divided by the number of cells
    accuracy = accuracy_df.sum().sum() / (accuracy_df.shape[0] * accuracy_df.shape[1])
    return round(accuracy, 4), accuracy_df

# calculate accuracy of random guesses
accuracy_random = get_accuracy_df(voted_docs_final_EP9, random_guess_EP9, 0.05)