import os
import requests
from typing import Any, Dict
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("openfda-server")

TIMEOUT = 20

def _get_api_key() -> str | None:
    return os.environ.get("OPENFDA_API_KEY")

def _query_openfda(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    api_key = _get_api_key()
    if api_key:
        params = dict(params)
        params["api_key"] = api_key
    
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # openFDA API returns a 404 HTTP status when no records match the query
        if e.response is not None and e.response.status_code == 404:
            return {"results": [], "info": "No matching records found"}
        
        status_code = e.response.status_code if e.response is not None else "Unknown"
        return {"error": f"HTTP error {status_code}: {str(e)}", "results": []}
    except requests.exceptions.Timeout as e:
        return {"error": f"Request timed out: {str(e)}", "results": []}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}", "results": []}

@mcp.tool()
def search_drug_labels(search: str, limit: int = 3, skip: int = 0) -> dict:
    """Search openFDA drug labels for warnings, contraindications, adverse reactions, and usage guidelines.

    Args:
        search: The search query, e.g., 'openfda.brand_name:"aspirin"'.
        limit: Max number of label records to return (default is 3).
        skip: Number of records to skip for pagination (default is 0).
    """
    url = "https://api.fda.gov/drug/label.json"
    params = {"search": search, "limit": limit, "skip": skip}
    return _query_openfda(url, params)

@mcp.tool()
def search_adverse_events(search: str, limit: int = 3, skip: int = 0) -> dict:
    """Search openFDA drug adverse event safety reports.

    Args:
        search: The search query, e.g., 'reactionmeddrapt:"headache"'.
        limit: Max number of event records to return (default is 3).
        skip: Number of records to skip for pagination (default is 0).
    """
    url = "https://api.fda.gov/drug/event.json"
    params = {"search": search, "limit": limit, "skip": skip}
    return _query_openfda(url, params)

if __name__ == "__main__":
    mcp.run(transport="stdio")
