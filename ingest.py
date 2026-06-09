import requests
import time
# from minsearch import Index
from sqlitesearch import TextSearchIndex



def fetch_documents():
    """Fetches course FAQ documents from the DataTalks.Club JSON API."""
    docs_url = "https://datatalks.club/faq/json/courses.json"
    response = requests.get(docs_url)
    courses_raw = response.json()

    documents = []
    url_prefix = "https://datatalks.club/faq"

    for course in courses_raw:
        course_url = f"{url_prefix}{course['path']}"
        course_response = requests.get(course_url)
        course_response.raise_for_status()
        course_data = course_response.json()
        documents.extend(course_data)
    
    return documents

def build_index(documents):
    """Initializes and fits the search index with the provided documents."""
    index = TextSearchIndex(
        text_fields=["question", "section", "answer"],  # full text search for matching keyword and also semantic search
        keyword_fields=["course"],          # Exact matching only
        db_path="faq.db"
    )

    # # add all at once
    # index.fit(documents)                   # Generated inverted index kind of DS for fast search operations

    # Simulate real time add one record at a time
    for doc in documents:
        index.add(doc)
        print(f"""Added: {doc["question"][:60]}...""")
        time.sleep(0.5)

    index.close()
    print("Done. Index saved to faq.db")

    
    print(f"Number of documents = {index.count}")
    return index

