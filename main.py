from dotenv import load_dotenv
from google import genai
from google.genai import types

from ingest import fetch_documents, build_index


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



def search(query, index):
    """Searches the index for relevant documents based on the query.
       Filtering - based on llm-zoomcamp course
       Ranking -  based on top K (or top 5) results
    """

    boost_dict = {'question': 2.0, 'section': 0.4}  
    filter_dict = {'course': 'llm-zoomcamp'}        

    # Get the Data source,, Used index instead of vector DB for now

    return index.search(
        query,
        boost_dict=boost_dict,      # Weights adjustments, more weight for question,  1 for 100%
        filter_dict=filter_dict,    # Filtering search (full match)
        num_results=5               # Ranking
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
    #return f"{INSTRUCTIONS.strip()}\n\n{prompt_body.strip()}"
    return f"{prompt_body.strip()}"

def llm(prompt, model="gemini-2.5-flash-lite"):
    """Sends the prompt to the Gemini model and returns the text response."""

    # response = client.models.generate_content(
    #     model="gemini-2.5-flash-lite",
    #     contents=prompt,
    # )

    response = client.models.generate_content(
        contents=prompt,
        model=model,
        # THE FIX: Isolate your system prompt rules here
        config=types.GenerateContentConfig(
            system_instruction=INSTRUCTIONS.strip()         # system_instruction is equivalant of developer role
        )
    )

    # Extract token metrics from the API response
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count

    # Calculate total cost based on Gemini 2.5 Flash-Lite pricing
    total_cost = (input_tokens * (0.10 / 1_000_000)) + (output_tokens * (0.40 / 1_000_000))
    print(f"API Cost for this call: ${total_cost:.8f}")


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
    index = build_index(documents)
    
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
