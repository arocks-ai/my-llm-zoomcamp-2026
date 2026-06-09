from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from ingest import fetch_documents, build_index

# Load environment variables (GEMINI_API_KEY)
load_dotenv()

# Initialize the Gemini client
client = genai.Client()

# # Moved to Ingest phase
# print("Initializing FAQ Assistant...")
# documents = fetch_documents()
# index = build_index(documents)





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

class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        course='llm-zoomcamp',
        model='gemini-2.5-flash-lite'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.course = course
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query):
        """Searches the index for relevant documents based on the query.
        Filtering - based on llm-zoomcamp course
        Ranking -  based on top K (or top 5) results
        """

        boost_dict = {'question': 3.0, 'section': 0.4}  
        filter_dict = {'course': self.course}



        #Check how many documents are in the index:
        print(f"Number of documents in the index: {self.index.count()}")
        search_results = self.index.search(query, boost_dict=boost_dict, filter_dict=filter_dict, num_results=5)

        # return self.index.search(
        #     query,
        #     boost_dict=boost_dict,      # Weights adjustments, more weight for question,  1 for 100%
        #     filter_dict=filter_dict,    # Filtering search (full match)
        #     num_results=5               # Ranking
        # )

        print("\n--- BEGIN - Search Top K Results from the datbase ---")
        for result in search_results:
            print("question = ",  result['question'])
            print("answer = ", result['answer'])
            print()
        print("\n--- END - Search Top K Results from the datbase ---\n\n")

        return search_results


    def build_context(self, search_results):
        """Converts search results into a formatted string for the LLM context."""
        lines = []
        for doc in search_results:
            lines.append(doc["section"])
            lines.append(f"Q: {doc['question']}")
            lines.append(f"A: {doc['answer']}")
            lines.append("")
        return "\n".join(lines).strip()

    def build_prompt(self, question, search_results):
        """Combines instructions, context, and question into a final prompt."""
        context = self.build_context(search_results)
        prompt_body = self.prompt_template.format(
            question=question,
            context=context
        )
        #return f"{INSTRUCTIONS.strip()}\n\n{prompt_body.strip()}"
        return f"{prompt_body.strip()}"

    def llm(self, prompt):
        """Sends the prompt to the Gemini model and returns the text response."""

        # response = client.models.generate_content(
        #     model="gemini-2.5-flash-lite",
        #     contents=prompt,
        # )

        response = client.models.generate_content(
            contents=prompt,
            model=self.model,
            # THE FIX: Isolate your system prompt rules here
            config=types.GenerateContentConfig(
                system_instruction=self.instructions.strip()         # system_instruction is equivalant of developer role
            )
        )

        # Extract token metrics from the API response
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count

        # Calculate total cost based on Gemini 2.5 Flash-Lite pricing
        total_cost = (input_tokens * (0.10 / 1_000_000)) + (output_tokens * (0.40 / 1_000_000))
        print(f"API Cost for this call: ${total_cost:.8f}")


        return response.text

    def rag(self, query):
        """Executes the complete Retrieval-Augmented Generation process."""

        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer

def connect_to_database():

    from sqlitesearch import TextSearchIndex

    db_name = "faq.db"
    file_path = Path(db_name)

    if not file_path.is_file():
        print(f"ERROR.. {db_name} was not found. Check if Ingestion phase was completed or not..")    
        exit

    # Connect to DB
    # Make sure ingestion is done before
    sqlite_index = TextSearchIndex(
        text_fields=["question", "section", "answer"],
        keyword_fields=["course"],
        db_path=db_name
    )

    return sqlite_index


def main():
    """Main entry point to initialize the system and run a query."""

    sqlite_index= connect_to_database()


    # query = "How to get the course completion certificate?"
    # query = "When does the next course starts"
    # query = "Do I get certification after completion"
    # query = "Are there any lectures/videos? Where are they?"
    # query = "can I use google ADK for this course"       # not in FAQ
    # query = "can I use Groq or ollama for this course"
    query = "How do  I run Olama locally for this course"
    print(f"\nUser Question: {query}")
    
    ragAssitant = RAGBase(sqlite_index, client) 
    answer = ragAssitant.rag(query)

    print("\n--- user Question and Final Answer ---")
    print(f"\nUser Question: {query}")
    print(f"Final Answer: {answer}")

if __name__ == "__main__":
    main()
