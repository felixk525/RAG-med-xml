#path = "D:\Bachelorarbeit\WikiQA\train.csv"
# questionID string, question string, document title string, answer string, label integer (1 = relevant, 0 = irrelevant)
# many answers are irrelevant!

import pandas as pd
import requests
import time
import pandas as pd
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
import json

link = "https://en.wikipedia.org/wiki/" # Wikipedias usual prefix url
robot_url = "https://en.wikipedia.org/robots.txt" # Wikipedias crawling guidelines
# Path to the dataset
path = "D:/Bachelorarbeit/WikiQA/train.csv" # The WikiQA file
output_file = "D:/Bachelorarbeit/WikiQA/crawled_data.jsonl" # The benchmark created
def main():
    with open(output_file, "w", encoding="utf-8") as f:
        pass
    read_data = pd.read_csv(path)
    checker = RobotsChecker(robot_url)
    relevant_data = read_data[read_data['label'] == 1] # Only rows with relevant answers
    data = relevant_data.drop_duplicates(subset='question_id') # Eliminate duplicate questions
    print(f"Number of unique relevant rows: {len(data)}")
    data = data.assign(wiki_link=link + data['document_title'].str.replace(' ', '_')) # replace whitespace with _
    data = data.reset_index(drop=True)
    crawled_links = crawl_links(data, output_file, checker)
    print(f"Successfully crawled {len(crawled_links)} of {len(data)} links.")

class RobotsChecker:
    """
    A class to manage and cache robots.txt data for a website.
    """
    def __init__(self, robots_url):
        self.rp = RobotFileParser()
        self.rp.set_url(robots_url)
        try:
            self.rp.read()
            self.valid = True
        except Exception as e:
            print(f"Could not read robots.txt from {robots_url}. Error: {e}")
            self.valid = False

    def is_allowed(self, url):
        """
        Input:
            url (str): The URL to check.
        Returns:
            bool: True if crawling is allowed, False otherwise.
        """
        if not self.valid:
            return False  # Default to not allowed if robots.txt could not be read
        return self.rp.can_fetch("*", url)  # "*" assumes no specific user-agent string


def fetch_html(url, checker):
    """
    Input:
        url (str): The URL to retrieve HTML data from.
    Returns:
        str: The HTML content of the page, or an empty string if the request fails.
    """
    try:
        # Check if crawling is allowed
        if not checker.is_allowed(url):
            print(f"Crawling disallowed by robots.txt: {url}")
            return ""
        
        # Send an HTTP GET request
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}. Entry was {url}")
            return ""
    except requests.RequestException as e:
        print(f"An error occurred while fetching {url}: {e}")
        return ""

def crawl_links(data, output_file, checker):
    """
    Crawler loop
    Input:
        data (pd.DataFrame): DataFrame containing a column 'wiki_link' with URLs.
        output_file (str): Path to save the retrieved HTML content in JSONL format.
    Returns:
        list: A list of all successfully crawled links.
    """
    crawled_links = []

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in data.iterrows():
            url = row['wiki_link']
            html_content = fetch_html(url, checker)
            if (idx % 20 == 0):
                print(f"Crawled {idx} of {len(data)} links.")
            
            if html_content:
                crawled_links.append(url)
                json_line = json.dumps({"question": row["question"], "html": html_content, "answer": row["answer"]}, ensure_ascii=False)
                f.write(json_line + "\n")  # Write each entry as a new line in the JSONL file
            
            # Pause between requests to be polite
            time.sleep(0.2)

    print(f"Saved HTML content to {output_file}")
    return crawled_links

if __name__ == "__main__":
    main()

# Results should be something like:
# Crawled 120 of 873 links.
# Failed to fetch data. Status code: 404. Entry was https://en.wikipedia.org/wiki/EstÃ©e_Lauder_Companies
# Crawled 140 of 873 links.
