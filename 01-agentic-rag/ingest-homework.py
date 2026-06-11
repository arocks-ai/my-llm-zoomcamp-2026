import requests
import time
# from minsearch import Index
from sqlitesearch import TextSearchIndex
from gitsource import GithubRepositoryDataReader

def fetch_documents():
    """Fetches course FAQ documents from the DataTalks.Club JSON API."""


    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()
    print("file count = ", len(files))

    # print first file name and contents
    file = files[0]
    print(f"file name = {file.filename}")
    print(f"file content = {file.parse()}")

    return files





# def fetch_documents():
#     """Fetches course FAQ documents from the DataTalks.Club JSON API."""
#     docs_url = "https://datatalks.club/faq/json/courses.json"
#     response = requests.get(docs_url)
#     courses_raw = response.json()

#     documents = []
#     url_prefix = "https://datatalks.club/faq"

#     for course in courses_raw:
#         course_url = f"{url_prefix}{course['path']}"
#         course_response = requests.get(course_url)
#         course_response.raise_for_status()
#         course_data = course_response.json()
#         documents.extend(course_data)
    
#     return documents

def build_index(files):
    """Initializes and fits the search index with the provided documents."""
    index = TextSearchIndex(
        text_fields=["content"],        # full text search for matching keyword and also semantic search
        keyword_fields=["filename"],    # Exact matching only
        db_path="contents.db"
    )

    # # File format will be like below
    # {
    #     "filename": "01-agentic-rag/lessons/03-rag.md",
    #     "content" : "contents of the file"
    # }


    documents = []
    for file in files:
        doc = {
            "filename": file.filename,
            "content": file.parse()['content']
        }
        documents.append(doc)


    # Simulate real time, add one record at a time
    for doc in documents:
        index.add(doc)
        print(f"""Added: {doc["content"][:60]}...""")
        time.sleep(0.1)

    print(f"Number of documents = {index.count}")
    index.close()
    print("Done. Index saved to the database")

    
    return index



def main():
    """Main entry point to initialize the system and run a query."""

    print("Prepare data by Creating DB with FAQ information ...")
    files = fetch_documents()
    index = build_index(files)

if __name__ == "__main__":
    main()

