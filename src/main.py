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
os.environ["AZURE_OPENAI_API_INSTANCE_NAME"] = os.getenv("AZURE_OPENAI_API_INSTANCE_NAME")
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

main_agent = create_deep_agent(
    name="MainAgent",
    system_prompt="You are the main agent responsible for coordinating tasks and managing sub-agents.",
    subagents=[research_agent],
    model=model)

if __name__ == "__main__":
    topic = "The impact of AI on modern healthcare"
    response = main_agent.invoke({"messages": [{"role": "user", "content": f"Conduct research on the following topic: {topic}"}]})
    print(response)
    print("Research Summary:")
    print(response)