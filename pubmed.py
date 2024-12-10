import os
import requests
import pickle
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import re

# PubMed API URLs
PUBMED_API_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Function to preprocess queries
def preprocess_query(query):
    """
    Preprocess the query string to handle variations in terms.
    E.g., `treat` will match `treat`, `treats`, `treated`, `treatment`, etc.
    """
    term_variations = {
        r'\btreat\b': r'(treat|treats|treated|treating|treatment|treatments)',
        r'\bcell line\b': r'(cell line|cell lines)',
    }

    for base_term, expanded_term in term_variations.items():
        query = re.sub(base_term, expanded_term, query, flags=re.IGNORECASE)

    return query

# Function to query PubMed and retrieve PMIDs
def search_pubmed(query, api_key, retmax=10):
    """Search for PubMed articles matching the query."""
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': retmax,
        'api_key': api_key,
    }
    response = requests.get(PUBMED_API_BASE, params=params)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    pmids = [id_tag.text for id_tag in root.findall(".//Id")]
    return pmids

# Function to fetch PubMed metadata, including DOIs and full-text links
def fetch_pubmed_links(pmid, api_key):
    """Fetch PubMed metadata, including full-text links."""
    params = {
        'db': 'pubmed',
        'id': pmid,
        'api_key': api_key,
        'retmode': 'xml'
    }
    response = requests.get(PUBMED_FETCH_API, params=params)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    full_text_links = []

    for article_link in root.findall(".//ELocationID"):
        if article_link.attrib.get("EIdType") == "doi":
            doi = article_link.text
            full_text_links.append(f"https://proxyiub.uits.iu.edu/login?url=https://doi.org/{doi}")

    return full_text_links

# Function to manually authenticate and save cookies
def authenticate_and_save_cookies(cookie_file="ezproxy_cookies.pkl"):
    """Manually authenticate via IU Proxy and save cookies."""
    proxy_url = "https://proxyiub.uits.iu.edu"
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=options)

    try:
        driver.get(proxy_url)
        print("Please log in manually and handle Duo 2FA. Waiting for completion...")
        time.sleep(60)  # Allow time for manual login
        cookies = driver.get_cookies()
        with open(cookie_file, "wb") as file:
            pickle.dump(cookies, file)
        print(f"Cookies saved to {cookie_file}.")
    finally:
        driver.quit()

# Function to fetch an article using saved cookies
def fetch_article_with_cookies(url, cookie_file="ezproxy_cookies.pkl"):
    """Fetch full-text article using saved cookies."""
    session = requests.Session()

    # Load cookies from file
    with open(cookie_file, "rb") as file:
        cookies = pickle.load(file)

    for cookie in cookies:
        session.cookies.set(cookie['name'], cookie['value'], domain=cookie['domain'])

    response = session.get(url, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    full_text = "\n".join([tag.get_text(separator="\n") for tag in soup.find_all(['p', 'div', 'section'])]).strip()

    if len(full_text) < 200:  # Validate content length
        print("Insufficient content. Check access or retry.")
        return None

    return full_text

# Function to save text to a specified output directory
def save_text_to_file(text, pmid, output_dir):
    """Save the textual content to a file in the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{pmid}_content.txt")
    with open(output_file, "w", encoding="utf-8") as text_file:
        text_file.write(text)
    print(f"Saved content to {output_file}.")

# Workflow to retrieve papers using PubMed
def retrieve_papers_with_pubmed(query, api_key, output_dir, max_results=10):
    """Retrieve full-text papers for a query using PubMed and IU Proxy."""
    # Preprocess query
    processed_query = preprocess_query(query)
    print(f"Processed Query: {processed_query}")

    # Search PubMed for PMIDs
    pmids = search_pubmed(processed_query, api_key, retmax=max_results)
    print(f"Found {len(pmids)} articles.")

    # Authenticate manually once and save cookies
    if not os.path.exists("ezproxy_cookies.pkl"):
        authenticate_and_save_cookies()

    for pmid in pmids:
        print(f"Processing PMID: {pmid}")

        # Fetch PubMed full-text links
        links = fetch_pubmed_links(pmid, api_key)
        if not links:
            print(f"No full-text links found for PMID {pmid}. Skipping.")
            continue

        # Use the first link (or handle multiple links)
        for link in links:
            print(f"Fetching article from: {link}")
            try:
                content = fetch_article_with_cookies(link)
                if content:
                    save_text_to_file(content, pmid, output_dir)
                    break  # Exit loop after successful retrieval
            except Exception as e:
                print(f"Failed to fetch article for PMID {pmid} from {link}: {e}")

    print("Finished processing all PMIDs.")

# Define the query
query = '"breast cancer" AND "cell line" AND "treat"'

# PubMed API key (replace with your actual key)
api_key = "PubmedKey"

# Specify output directory
output_dir = "/Users/leasyrin/Downloads/IUB_PhD/CS/L645:B659/Final project/pubmed_queries/full_outputs"

# Retrieve full-text papers
retrieve_papers_with_pubmed(query, api_key, output_dir, max_results=50000)
