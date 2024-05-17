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

def get_majorities(df, threshold):
    """
    Takes a DataFrame and a threshold. The threshold defines which percentage of votes
    of biggest left- and biggest right-leaning party group is needed to be considered a 
    left-leaning or right-leaning majority.
    Returns a DataFrame with columns for each majority type (Consensus, RM, LM) and a
    binary value for each vote indicating if it is part of the majority.
    """
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
    """
    Takes a DataFrame and a list of thresholds. Returns a DataFrame with the number of votes 
    that are regarded as a majority vote when using a specific threshold to define majority.
    Dataframe includes the sum of votes and percentage of votes regarded majority for each 
    threshold.
    """
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

        perc_Con = round(sum_Consensus/df.shape[0], 2)
        perc_RM = round(sum_RM/df.shape[0], 2)
        perc_LM = round(sum_LM/df.shape[0], 2)

        # Store the results in the DataFrame under the current threshold
        threshold_df.loc[item] = [sum_Consensus, perc_Con, sum_RM, perc_RM, sum_LM, perc_LM]
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
    df['Interinstitutional file number'] = str_parts.str[0] + "/" + str_parts.str[1] + '(' + str_parts.str[2] + ')'
    # create new column for report links
    df['Report link'] = base_url + df_copy['Interinstitutional file number'].astype(str)
    return df

def fetch_summary_link(url, session):
    """
    Fetches HTML content for a given URL, parses it, and extracts the summary link.
    """
    try:
        # Get HTML from the URL using a session object for better performance
        response = session.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the "button" element with id "summary"
        button = soup.find("button", {"id": "summary"})
        if button and 'onclick' in button.attrs:
            # Extract the link directly from the onclick attribute
            onclick_content = button['onclick'].split("'")[1]
            summary_link = "https://oeil.secure.europarl.europa.eu" + onclick_content
            #print(f"Extracted summary link: {summary_link}")
            return summary_link
        return None
    except requests.RequestException as e:
        print(f"Failed to retrieve or parse {url}: {e}")
        return "NA"

def extract_summary_links(urls):
    """
    Processes a list of URLs in parallel to extract summary links.
    Source: https://docs.python.org/3/library/concurrent.futures.html, Example
    """
    summary_links = []
    # Use requests.Session for connection pooling
    with requests.Session() as session:
        # Use ThreadPoolExecutor to execute requests in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Map fetch_summary_link function over all URLs with the session passed
            future_to_url = {executor.submit(fetch_summary_link, url, session): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if result:
                        summary_links.append(result)
                        print(f"Links extracted: {len(summary_links)}", end='\r')
                except Exception as exc:
                    print(f'{url} generated an exception: {exc}')
                    summary_links.append("NA")
    # Print final count on a new line once all URLs are processed
    print("\nTotal links extracted:", len(summary_links))
    return summary_links


def fetch_summary_texts(url, session):
    """
    Fetch the content of a URL and process the HTML to extract cleaned summary text.
    """
    try:
        response = session.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
    # Find the element using a regex to flexibly match the string
        summary_elements = soup.find_all('span', lang="EN-GB")
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
    # Create a session object
    with requests.Session() as session:
        # Use ThreadPoolExecutor to execute requests in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            # executor class should only be used via subclass such as map
            # Map lets fetch_and_process function be executed asynchronously with several calls in parallel
            future_to_url = {executor.submit(fetch_summary_texts, url, session): url for url in urls}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    results = future.result()
                    if results:
                        summaries.append(results)
                        print(f"Summaries extracted: {len(summaries)}", end='\r')
                    else:
                        summaries.append("NA")
                except Exception as exc:
                        print(f'{url} generated an exception: {exc}')
    return summaries

# def extract_summary_links_old(urls):
#     """
#     Takes each given link, parses html, finds "button" with id "summary", and combines 
#     summary link with base url to create a list of summary links.
#     """
#     summary_links = []
#     for url in urls:
#         # get html from url
#         response = requests.get(url).text
#         # parse html
#         soup = BeautifulSoup(response, 'html.parser')
#         # find all "button" elements with id "summary"
#         button = soup.find("button", {"id": "summary"})
#         # extract the link from the "data-url" attribute
#         onclick_content = [button[attr].split("'")[1] for attr in button.attrs if 'onclick' in attr]
#         summary_link = "https://oeil.secure.europarl.europa.eu" + str(onclick_content[0])
#         #print(summary_link)
#         summary_links.append(summary_link)
#         print(f"Links extracted: {len(summary_links)}", end='\r')
#         time.sleep(1)  # added to make the updating visible if loop is fast
#     # Print final count on a new line once all URLs are processed
#     print(f"Total links extracted: {len(summary_links)}")
#     return summary_links

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