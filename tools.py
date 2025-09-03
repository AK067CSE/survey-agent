# tools.py

from duckduckgo_search import DDGS

def simple_web_search(query: str, num_results: int = 5) -> str:
    """
    Performs a simple web search using DuckDuckGo and returns a formatted string of results.
    """
    print(f"--- Performing web search for: {query} ---")
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=num_results)]
            if not results:
                return "No search results found."
            
            # Format the results into a single string
            formatted_results = ""
            for result in results:
                formatted_results += f"Title: {result.get('title', 'N/A')}\n"
                formatted_results += f"Link: {result.get('href', 'N/A')}\n"
                formatted_results += f"Snippet: {result.get('body', 'N/A')}\n\n"
            
            return formatted_results
    except Exception as e:
        print(f"Error during web search: {e}")
        return f"An error occurred during the web search: {e}"
