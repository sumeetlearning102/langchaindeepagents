import os
from dotenv import load_dotenv
from typing import Literal
from langchain.chat_models import init_chat_model
from tavily import TavilyClient
from deepagents import create_deep_agent

# Load environment variables from .env file
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
os.environ["AZURE_OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")


def doInternetSearch(query:str, topic:Literal['general', 'news', 'finance'] = 'general'):
    """
    Perform an internet search using Tavily API
    
    Args:
        query: The search query string
        topic: The topic category for the search. Must be 'general', 'news', or 'finance'. Defaults to 'general'.
    
    Returns:
        Search results from Tavily including relevant articles and summaries
    """
    search_results = tavily_client.search(query=query, topic=topic, max_results=5)
    return search_results

research_agent_prompt = """You are a research agent that uses internet search to gather information on a given topic.
Your task is to perform searches and compile the results into a coherent summary.

1. Start with a clear research question.
2. Use the doInternetSearch function to find relevant information.
3. Summarize the findings in a structured format.
4. Provide citations for the sources used.

Good luck!
"""
research_agent = {
    "name": "ResearchAgent",
    "system_prompt": research_agent_prompt,
    "description": "An agent that performs internet research on a given topic.",
    "tools": [doInternetSearch]
}

model = init_chat_model("azure_openai:gpt-4o")

def parse_messages(result: dict):
    data = {
        "human": [],
        "ai": [],
        "tools": [],
    }

    for msg in result.get("messages", []):
        msg_type = type(msg).__name__
        #print(f"Processing message type: {msg_type}")
        
        # Check message type using type().__name__ instead of hasattr
        if msg_type == "ToolMessage":
            #print("ToolMessage detected")
            data["tools"].append({
                "tool_name": getattr(msg, "name", None),
                "tool_call_id": getattr(msg, "tool_call_id", None),
                "content": getattr(msg, "content", None)
            })
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            #print("AIMessage with tool calls detected")
            # AIMessage with tool calls (the reasoning + tool actions)
            calls = []
            for call in msg.tool_calls:
                calls.append({
                    "name": call["name"],
                    "args": call["args"]
                })
            data["ai"].append({
                "content": getattr(msg, "content", None),
                "tool_calls": calls
            })
        elif msg_type == "HumanMessage":
            #print("HumanMessage detected")
            data["human"].append({
                "type": msg_type,
                "content": getattr(msg, "content", None)
            })
        else:
            # Simple AIMessage without tool calls
            #print("AIMessage detected")
            data["ai"].append({
                "type": msg_type,
                "content": getattr(msg, "content", None)
            })
        #print(f"Current parsed data: {data}")
    return data

main_agent = create_deep_agent(
    name="MainAgent",
    system_prompt="You are the main agent responsible for coordinating tasks and managing sub-agents.",
    subagents=[research_agent],
    model=model)

if __name__ == "__main__":
    topic = "The impact of AI on modern healthcare"
    response = main_agent.invoke({"messages": [{"role": "user", "content": f"Conduct research on the following topic: {topic}"}]})
    print(response)
    # Save generated files to disk
    if "files" in response:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        for file_path, file_data in response["files"].items():
            # Clean up the file path (remove leading /)
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename)
            
            # Write content to file
            with open(output_path, "w", encoding="utf-8") as f:
                if isinstance(file_data.get("content"), list):
                    f.write("\n".join(file_data["content"]))
                else:
                    f.write(str(file_data.get("content", "")))
            
            print(f"Saved research file to: {output_path}")
    
    print("\nResearch Summary:")
    print(parse_messages(response))