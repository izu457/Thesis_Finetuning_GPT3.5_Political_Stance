# Description: This file contains the helper functions for the main script.

# Importing the necessary libraries
import csv
import os
import sys
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
import re
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


def read_xlsx(file):
    """
    Reads an Excel file and returns a DataFrame.
    """
    df = pd.read_excel(file, header=0)
    return df

def count_procedure(df):
    """
    Takes a DataFrame and returns a Series with the percentage of votes accepted or rejected per reading.
    """
    votes_first_reading_perc = df[df['Procedure'] == '***I'].shape[0]/df.shape[0]
    print(f"Between 2019-2022, {votes_first_reading_perc:.2%} of votes were accepted or rejected after the first reading.")
    votes_second_reading_perc = df[df['Procedure'] == '***II'].shape[0]/df.shape[0]
    print(f"Between 2019-2022, {votes_second_reading_perc:.2%} of votes were accepted or rejected after the second reading.")
    votes_third_reading_perc = df[df['Procedure'] == '***III'].shape[0]/df.shape[0]
    print(f"Between 2019-2022, {votes_third_reading_perc:.2%} of votes were accepted or rejected after the third reading.")
    votes_other_perc = df[df['Procedure'] == '***'].shape[0]/df.shape[0]
    print(f"Between 2019-2022, {votes_other_perc:.2%} of votes were on resolutions or motions of resolutions initiated by the parliament.")
    # return series with percentage of votes for each procedure
    return pd.Series([votes_first_reading_perc, votes_second_reading_perc, votes_third_reading_perc, votes_other_perc])

def party_abbr(df):
    """
    Takes a DataFrame and returns a DataFrame with aligned party group abbreviations.
    """
    df['EPG'] = df['EPG'].replace('Group of the European People\'s Party (Christian Democrats)', 'EPP')
    df['EPG'] = df['EPG'].replace('European Conservatives and Reformists Group', 'ECR')
    df['EPG'] = df['EPG'].replace('Group of the Greens/European Free Alliance', 'Greens/EFA')
    df['EPG'] = df['EPG'].replace('Group of the Alliance of Liberals and Democrats for Europe', 'REG')
    df['EPG'] = df['EPG'].replace('Group of the Progressive Alliance of Socialists and Democrats in the European Parliament', 'S&D')
    df['EPG'] = df['EPG'].replace('Non-attached Members', 'NI')
    df['EPG'] = df['EPG'].replace('Confederal Group of the European United Left - Nordic Green Left', 'The Left')
    df['EPG'] = df['EPG'].replace('Europe of Freedom and Direct Democracy Group', 'EFD/IDG')
    df['EPG'] = df['EPG'].replace('Europe of freedom and democracy Group', 'EFD/IDG')
    df['EPG'] = df['EPG'].replace('Europe of Nations and Freedom Group', 'EFD/IDG')
    df['EPG'] = df['EPG'].replace('IDG', 'EFD/IDG')
    return df

def calculate_percentage_votes(meps_grouped_df, group_members_df, period):
    """
    Takes a dataframe with grouped MEPs and a dataframe with the number of members in each group and period.
    Calculates the sum and percentage of votes in favour for each group.
    """
    # Create a copy of the dataframe to avoid modifying the original data
    df = meps_grouped_df.copy()
    # Map the number of members to each row using the group column
    df['members'] = df['EPG'].map(group_members_df[period])
    # Calculate 'Sum_in_favour' as integer
    df['Sum_in_favour'] = df['Vote'].astype(int)
    # Calculate the percentage by the number of members in the party found in group_members
    df['Perc_in_favour'] = (df['Vote'] / df['members']).round(3).astype(float)
    return df

def get_majorities(df, threshold):
    """
    Takes a DataFrame and a threshold. The threshold defines which percentage of votes
    of biggest left- and biggest right-leaning party group is needed to be considered a 
    left-leaning or right-leaning majority.
    Returns a DataFrame with columns for each majority type (General, LM, RM) 
    and a binary value for each vote indicating if it is part of the majority.
    """
    # Create new DataFrame columns as calculations (not modifying the original df)
    majority = (df['ECR%'] >= threshold) & (df['S&D%'] >= threshold)
    rm = (df['ECR%'] >= threshold) & (df['EFD/IDG%'] >= threshold) & (df['EPP%'] >= threshold) & (df['Greens/EFA%'] < threshold) & (df['The Left%'] < threshold)
    lm = (df['ECR%'] < threshold) & (df['EFD/IDG%'] < threshold) & (df['S&D%'] >= threshold) & (df['Greens/EFA%'] >= threshold) & (df['The Left%'] >= threshold)

    # Return the results as a new DataFrame
    result = pd.DataFrame(index=df.index, data={
        'General Majority': majority.astype(int),
        'Right Majority': rm.astype(int),
        'Left Majority': lm.astype(int)
    })
    return result

def test_thresholds(df, thresholds):
    """
    Takes a DataFrame and a list of thresholds. Returns a DataFrame with the number of votes 
    that are regarded as a majority vote when using a specific threshold to define majority.
    Dataframe includes the sum of votes and percentage of votes regarded majority for each 
    threshold.
    """
    # Prepare a DataFrame to store the results
    threshold_df = pd.DataFrame(index=thresholds, columns=['General Majority', "GM%", 'Right Majority', "RM%", 'Left Majority', "LM%"])

    # Create a deep copy of the DataFrame to avoid any mutation
    df_copy = df.copy(deep=True)

    for item in thresholds:
        # Apply the get_majorities to get the counts
        result_df = get_majorities(df_copy, item)
        sum_Majority = result_df['General Majority'].sum()
        sum_RM = result_df['Right Majority'].sum()
        sum_LM = result_df['Left Majority'].sum()

        perc_Con = round(sum_Majority/df.shape[0], 2)
        perc_RM = round(sum_RM/df.shape[0], 2)
        perc_LM = round(sum_LM/df.shape[0], 2)

        # Store the results in the DataFrame under the current threshold
        threshold_df.loc[item] = [sum_Majority, perc_Con, sum_RM, perc_RM, sum_LM, perc_LM]
    # rename index
    threshold_df.index.name = 'Threshold'
    return threshold_df

def get_vote_alingment(df, groups, majorities):
    """
    Takes a DataFrame, a list of party groups, and a list of majorities to compare with.
    Returns a DataFrame with the number of votes of each party groups that aligned with 
    each majority ("How many of group x voted with a left majority?"), and the percentage
    of votes that align with each majority for each group.
    """
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

def construct_report_links(df):
    """
    From the given dataframe, takes the value of the column "Interinstitutional file number",
    converts it to a string, and appends it to the base url to create a link to the report.
    Returns a DataFrame with the new column "Report link" containing the links.
    """
    df_copy = df.copy()
    # take value of column "Interinsitutional file number", convert it to string, append it to base url
    base_url = "https://oeil.secure.europarl.europa.eu/oeil/popups/ficheprocedure.do?lang=en&reference="
    # split string at second / and replace with (
    str_parts = df_copy['Interinstitutional file number'].str.split("/")
    df_copy['Interinstitutional file number'] = str_parts.str[0] + "/" + str_parts.str[1] + '(' + str_parts.str[2] + ')'
    # create new column for report links
    df['Report link'] = base_url + df_copy['Interinstitutional file number'].astype(str)
    return df

def fetch_summary_link(url):
    """
    Fetches HTML content for a given URL, parses it, and extracts the summary link.
    """
    try:
        # Get HTML from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find all "button" elements with id "summary"
        buttons = soup.find_all("button", {"id": "summary"})
        # Iterate over found buttons to find the one with the specific title
        #for button in buttons:
        #    if button.get('title') == "Summary for Legislative proposal published":
        #        print(button)
        for button in buttons:
            if (
                button.get('title') == "Summary for Legislative proposal published"
                ) or (
                    "Summary for Committee report tabled for plenary, 1st reading" in button.get('title')
                      ) or (
                          button.get('title') == "Summary for Preparatory document"
                          ):
                # Check if the button has an 'onclick' attribute
                if 'onclick' in button.attrs:
                    # Extract the link directly from the onclick attribute
                    onclick_content = button['onclick'].split("'")[1]
                    summary_link = "https://oeil.secure.europarl.europa.eu" + onclick_content
                    return summary_link
            elif "Summary for Decision by Parliament" in button.get('title'):
                # Check if the button has an 'onclick' attribute
                if 'onclick' in button.attrs:
                    # Extract the link directly from the onclick attribute
                    onclick_content = button['onclick'].split("'")[1]
                    summary_link = "https://oeil.secure.europarl.europa.eu" + onclick_content
                    return summary_link
        return None  # Return None if no matching button is found
    except requests.RequestException as e:
        print(f"Failed to retrieve or parse {url}: {e}")
        return "NA"
    
def extract_summary_links(urls):
    """
    Processes a list of URLs sequentially to extract summary links.
    """
    summary_links = []
    for url in urls:
        try:
            result = fetch_summary_link(url)
            if result:
                summary_links.append(result)
                print(f"Links requested: {len(summary_links)}", end='\r')
            else:
                summary_links.append("NA")
        except Exception as exc:
            print(f'{url} generated an exception: {exc}')
            summary_links.append("NA")
    # Print final count on a new line once all URLs are processed
    print(f"\nTotal links successfully extracted {len(summary_links) - summary_links.count("NA")} out of {len(summary_links)}")
    return summary_links
    
def fetch_summary_texts(url):
    """
    Fetch the content of a URL and process the HTML to extract cleaned summary text.
    Two types of encoding of text in htmls are considered.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        # type 1 of htmls
        summary_elements = soup.find_all('span', lang="EN-GB")
        # if hyperlink in summary, extract text from hyperlink
        if summary_elements:
            summary_text = " ".join(re.sub(r'\s+', ' ', item.text.strip()) for item in summary_elements)
            return summary_text
        # type 2 of htmls
        summary_elements = soup.find_all('p', style="text-align: justify;")
        if summary_elements:
            summary_text = " ".join(re.sub(r'\s+', ' ', item.text.strip()) for item in summary_elements)
            return summary_text
        return None
    except requests.RequestException as e:
        print(f"Failed to retrieve or parse {url}: {e}")
        return "NA"
    
def extract_summary_texts(urls):
    """
    Takes a list of urls and extracts the summary text from the html of the page
    by combining all the paragraphs of the summary into one string.
    Returns a list of summary texts.
    """
    summaries = []
    for url in urls:
        try:
            result = fetch_summary_texts(url)
            if result:
                summaries.append(result)
                print(f"Summaries extracted: {len(summaries)}", end='\r')
            else:
                summaries.append("NA")
        except Exception as exc:
            print(f'{url} generated an exception: {exc}')
    # Print final count on a new line once all URLs are processed
    print(f"\nTotal summaries successfully extracted {len(summaries) - summaries.count("NA")} out of {len(summaries)}")
    return summaries

def get_random_guess(data):
    """
    Creates a random guess for the group percentages and majorities for each vote.
    Returns dataframe with random guesses.
    """
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
    # try to set index of test dataframe to Vote ID if not already done
    try:
        test_df.set_index('Vote ID', inplace=True)
    except:
        pass
    # create a new dataframe with the same index as the original dataframe
    accuracy_df = pd.DataFrame(index=test_df.index)
    # add a column for each column in the original dataframe
    for column in test_df.columns:
        accuracy_df[column] = 0
    # loop through the rows of the test dataframe
    for index in test_df.index:
        # loop through the columns of the test dataframe
        for column in test_df.columns:
            # Debugging: Print the types and values
            #print(f"Index: {index}, Column: {column}")
            #print(f"Original DF Value: {original_df.loc[index, column]}, Type: {type(original_df.loc[index, column])}")
            #print(f"Test DF Value: {test_df.loc[index, column]}, Type: {type(test_df.loc[index, column])}")
            
            # check if the value in the test dataframe is within the tolerance interval of the original dataframe
            if ((original_df.loc[index, column] - tolerance_interval) <= test_df.loc[index, column]) & (test_df.loc[index, column] <= (original_df.loc[index, column] + tolerance_interval)):                # if it is, add 1 to the corresponding cell in the accuracy dataframe
                accuracy_df.loc[index, column] = 1
    # calculate the accuracy as the sum of the accuracy dataframe divided by the number of cells
    accuracy = accuracy_df.sum().sum() / (accuracy_df.shape[0] * accuracy_df.shape[1])
    return round(accuracy, 4), accuracy_df

def process_session(voted_docs, party_names, majority_names):
    """
    Combines the functions get_random_guess and get_accuracy_df to calculate the accuracy 
    of a random guess for party and majority votes individually.
    Takes as arguents a dataframe of voted documents and the names of the columns for the 
    party and majority votes.
    Returns the accuracy for party and majority votes.
    """
    random_guess = get_random_guess(voted_docs)
    random_party = get_accuracy_df(voted_docs, random_guess[party_names], 0.05)
    random_majority = get_accuracy_df(voted_docs, random_guess[majority_names], 0.05)
    return random_party[0], random_majority[0]

def split_select_data(data, training_fraction):
    """"
    Function to split the data into training and testing and to drop unnecessary columns.
    Takes as arguments the data and the fraction of the data to be used for training.
    Returns the training and testing data.
    """
    # create a copy of the data
    data = data.copy()
    # drop unnecessary columns
    data = data[["Title",  "ECR%", "EPP%", "Greens/EFA%","EFD/IDG%",
                 "NI%","REG%","S&D%", "The Left%", "General Majority", 
                 "Left Majority", "Right Majority", "Consensus", "Summary text"
                ]]
    # sample the training data
    training = data.sample(frac=training_fraction)
    # the rest of the data is the testing data
    testing = data.drop(training.index)
    print(training.shape)
    print(testing.shape)
    return training, testing