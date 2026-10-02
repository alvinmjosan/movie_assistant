import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import sys

# Connect to other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingestion.vector_store import query_vector_store
from mcp.email_tool import send_email

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define the GPT Model
# Note: We use gpt-4o-mini because it perfectly supports function calling and is very cost effective.
GPT_MODEL = "gpt-4o-mini"

def get_agent_response(user_query: str, chat_history: list = None) -> str:
    """
    The core agent that decides whether to answer directly, ask for context,
    query the RAG database, or trigger the email tool.
    """
    if chat_history is None:
        chat_history = []
        
    # The tools (functions) that GPT is allowed to call autonomously
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_movie_database",
                "description": "Searches the vector database for actual movie dialogue/quotes/scenes. Use this whenever the user asks a question about a movie or character.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The specific topic or quote to search for in the database.",
                        }
                    },
                    "required": ["query"],
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "send_email",
                "description": "Sends an email to a recipient. DO NOT GUESS the recipient's email. If the user doesn't provide it, clarify with them first.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient_email": {
                            "type": "string",
                            "description": "The email address to send to.",
                        },
                        "subject": {
                            "type": "string",
                            "description": "The subject of the email.",
                        },
                        "body": {
                            "type": "string",
                            "description": "The body of the email. It should include movie quotes and citations if relevant.",
                        }
                    },
                    "required": ["recipient_email", "subject", "body"],
                },
            }
        }
    ]

    # Instruct the agent on its core parameters
    system_prompt = {
        "role": "system",
        "content": (
            "You are a professional Movie Intelligence AI. "
            "Your job is to answer questions using movie metadata and dispatch emails when requested. "
            "CRITICAL RULES: "
            "1. If a user asks a factual question, ALWAYS use the query_movie_database tool. "
            "2. When providing movie facts, ALWAYS include the citation (Movie Title, Timestamp Range). "
            "3. If a user asks to send an email, verify you have the recipient's email address. If it is vague or missing (e.g. 'email John text'), YOU MUST ask the user to clarify the exact email address before taking action! Do NOT guess."
        )
    }
    
    messages = [system_prompt] + chat_history + [{"role": "user", "content": user_query}]
    
    # 1. Ask GPT what to do
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # 2. Check if GPT decided it needs to use a tool
    if tool_calls:
        # We append GPT's invisible tool-call request to the conversation logic
        messages.append(response_message)
        
        # 3. Execute the tools locally
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"--> [System] Agent requested execution of '{function_name}' with args {function_args}")
            
            if function_name == "query_movie_database":
                db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ingestion", "chroma_db")
                search_results = query_vector_store(function_args.get("query"), persist_directory=db_path)
                
                # Format the messy array context into a clean string for GPT
                docs = search_results.get("documents", [[]])[0]
                metas = search_results.get("metadatas", [[]])[0]
                
                context_str = ""
                for i in range(len(docs)):
                    m = metas[i]
                    context_str += f"- Dialogue: {docs[i]}\n  Citation: [{m['movie_title']}, {m['start_time']} - {m['end_time']}]\n"
                
                function_response = context_str if context_str else "No relevant data found in the database."
                
            elif function_name == "send_email":
                function_response = send_email(
                    recipient_email=function_args.get("recipient_email"),
                    subject=function_args.get("subject"),
                    body=function_args.get("body")
                )
                
            # Send the result of the tool run back to GPT
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })
            
        # 4. Get the final response from GPT now that it has the tool data
        second_response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
        )
        return second_response.choices[0].message.content
        
    else:
        # GPT did not need a tool (e.g. asking clarifying question or chatting normally)
        return response_message.content

if __name__ == "__main__":
    # Small terminal testing harness
    print("Agent is ready. (Type 'quit' to exit)")
    history = []
    while True:
        q = input("\nYou: ")
        if q.lower() in ("quit", "exit"):
            break
        print("Agent thinking...")
        ans = get_agent_response(q, history)
        print(f"\nAgent: {ans}")
        # Keep recent history
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": ans})
