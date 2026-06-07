import os
import requests
import json
from dotenv import load_dotenv
from google import genai
from minsearch import Index

# Load environment variables (GEMINI_API_KEY)
load_dotenv()

# Initialize the Gemini client
client = genai.Client()

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""

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

def setup_index(documents):
    """Initializes and fits the search index with the provided documents."""
    index = Index(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"]
    )
    index.fit(documents)
    return index

def search(query, index):
    """Searches the index for relevant documents based on the query."""
    boost_dict = {'question': 2.0, 'section': 0.4}
    filter_dict = {'course': 'llm-zoomcamp'}

    return index.search(
        query,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
        num_results=5
    )

def build_context(search_results):
    """Converts search results into a formatted string for the LLM context."""
    lines = []
    for doc in search_results:
        lines.append(doc["section"])
        lines.append(f"Q: {doc['question']}")
        lines.append(f"A: {doc['answer']}")
        lines.append("")
    return "\n".join(lines).strip()

def build_prompt(question, search_results):
    """Combines instructions, context, and question into a final prompt."""
    context = build_context(search_results)
    prompt_body = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )
    return f"{INSTRUCTIONS.strip()}\n\n{prompt_body.strip()}"

def llm(prompt):
    """Sends the prompt to the Gemini model and returns the text response."""
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    return response.text

def rag(query, index):
    """Executes the complete Retrieval-Augmented Generation process."""
    search_results = search(query, index)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

def main():
    """Main entry point to initialize the system and run a query."""
    print("Initializing FAQ Assistant...")
    documents = fetch_documents()
    index = setup_index(documents)
    
    # query = "How to get the course completion certificate?"
    # query = "When does the next course starts"
    # query = "Do I get certification after completion"
    # query = "Are there any lectures/videos? Where are they?"
    # query = "can I use google ADK for this course"       # not in FAQ
    query = "can I use Groq or ollama for this course"
    print(f"\nUser Question: {query}")
    
    answer = rag(query, index)
    
    print("\n--- Answer ---")
    print(answer)

if __name__ == "__main__":
    main()
