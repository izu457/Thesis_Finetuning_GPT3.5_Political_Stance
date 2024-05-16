# Description: This file contains the helper functions for the main script.

# Importing the necessary libraries
import csv
import os
import sys
import pandas as pd

def get_majorities(df, threshold):
    # Create new DataFrame columns as calculations (not modifying the original df)
    consensus = (df['EPP%'] >= threshold) & (df['S&D%'] >= threshold)
    rm = (df['EPP%'] >= threshold) & (df['S&D%'] < threshold)
    lm = (df['S&D%'] >= threshold) & (df['EPP%'] < threshold)

    # Return the results as a new DataFrame
    result = pd.DataFrame(index=df.index, data={
        'Consensus': consensus.astype(int),
        'RM': rm.astype(int),
        'LM': lm.astype(int)
    })
    return result

def test_thresholds(df, thresholds):
    # Prepare a DataFrame to store the results
    threshold_df = pd.DataFrame(index=thresholds, columns=['Consensus', "Consensus%", 'RM', "RM%", 'LM', "LM%"])

    # Create a deep copy of the DataFrame to avoid any mutation
    df_copy = df.copy(deep=True)

    for item in thresholds:
        # Apply the get_majorities to get the counts
        result_df = get_majorities(df_copy, item)
        sum_Consensus = result_df['Consensus'].sum()
        sum_RM = result_df['RM'].sum()
        sum_LM = result_df['LM'].sum()

        perc_Con = round(sum_Consensus/total_votes_leg, 2)
        perc_RM = round(sum_RM/total_votes_leg, 2)
        perc_LM = round(sum_LM/total_votes_leg, 2)

        # Store the results in the DataFrame under the current threshold
        threshold_df.loc[item] = [sum_Consensus, perc_Con, sum_RM, perc_RM, sum_LM, perc_LM]
    # rename index
    threshold_df.index.name = 'Threshold'
    return threshold_df

def get_vote_alingment(df, groups, majorities):
    # create empty dataframe of shape (len(groups), len(majorities))
    maj_df = pd.DataFrame(index=groups, columns=[majorities*2])
    for group in groups:
        group_val = []
        for maj in majorities:
            sum_maj_group = df[(df[maj] == 1) & (df[group] >= 0.66)].shape[0]
            perc_maj_group = round(sum_maj_group/df[maj].sum(), 4)
            formatted_perc = "{:.2f}%".format(perc_maj_group * 100)
            group_val.append(sum_maj_group)
            group_val.append(formatted_perc)
        # add group_val as row to maj_df
        maj_df.loc[group] = group_val
        # rename columns
        maj_df.index.name = 'Voted with'

    return maj_df

# def get_majorities_old(df, threshold):
#     for index, row in df.iterrows():
#         if row["EPP"] >= threshold and row["S&D"] >= threshold:
#             df.loc[index, 'Consensus'] = 1
#             df.loc[index, 'RM'] = 0
#             df.loc[index, 'LM'] = 0
#         elif row["S&D"] >= threshold:
#             df.loc[index, 'LM'] = 1
#             df.loc[index, 'RM'] = 0
#         elif row["EPP"] >= threshold:
#             df.loc[index, 'RM'] = 1
#             df.loc[index, 'LM'] = 0
#         else:
#             df.loc[index, 'RM'] = 0
#             df.loc[index, 'LM'] = 0
#             df.loc[index, 'Consensus'] = 0
#     return df

# def test_thresholds_old(df, thresholds):
#     # set up empty dataframe to store thresholds
#     threshold_df = pd.DataFrame(index=thresholds, columns=['Consensus', 'RM', 'LM'])
#     for threshold in thresholds:
#         df_copy = df.copy()
#         print(f"Before get_majorities for threshold {threshold}:")

#         df_processed = get_majorities(df_copy, threshold)
        
#         print(f"After get_majorities for threshold {threshold}:")

#         sum_Consensus = df_processed['Consensus'].sum()
#         sum_RM = df_processed['RM'].sum()
#         sum_LM = df_processed['LM'].sum()

#         # Set the results for this threshold in the dataframe
#         threshold_df.loc[threshold] = [sum_Consensus, sum_RM, sum_LM]


#     return threshold_df