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

def get_majorities(df, threshold):
    """
    Takes a DataFrame and a threshold. The threshold defines which percentage of votes
    of biggest left- and biggest right-leaning party group is needed to be considered a 
    left-leaning or right-leaning majority.
    Returns a DataFrame with columns for each majority type (Consensus, RM, LM) and a
    binary value for each vote indicating if it is part of the majority.
    """
    # Create new DataFrame columns as calculations (not modifying the original df)
    majority = (df['EPP%'] >= threshold) & (df['S&D%'] >= threshold)
    rm = (df['EPP%'] >= threshold) & (df['S&D%'] < threshold)
    lm = (df['S&D%'] >= threshold) & (df['EPP%'] < threshold)

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
        # print number of elements in button
        print(f"Number of elements in button: {len(button)}")
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
                    else:
                        summary_links.append("NA")
                except Exception as exc:
                    print(f'{url} generated an exception: {exc}')
                    summary_links.append("NA")
    # Print final count on a new line once all URLs are processed
    print("\nTotal links extracted:", len(summary_links))
    return summary_links
    
def fetch_summary_texts(url):
    """
    Fetch the content of a URL and process the HTML to extract cleaned summary text.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        print(soup)
    # Find the element using a regex to flexibly match the string
        summary_elements = soup.find_all('span', lang="EN-GB")
        # if hyperlink in summary, extract text from hyperlink
        print(summary_elements)

        if summary_elements:
            summary_text = " ".join(re.sub(r'\s+', ' ', item.text.strip()) for item in summary_elements)
            return summary_text
        return None
    except requests.RequestException as e:
        print(f"Failed to retrieve or parse {url}: {e}")
        return "NA"

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
