import logging
import httpx
from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rxnorm_server")

# Initialize FastMCP server
mcp = FastMCP("RxNorm")

BASE_URL = "https://rxnav.nlm.nih.gov/REST"

@mcp.tool()
async def search_drug(name: str, max_entries: int = 5) -> List[Dict[str, Any]]:
    """
    Search for a drug by name, handling misspellings and returning candidates with confidence scores.
    Uses the RxNorm approximateTerm endpoint.
    
    Args:
        name: The drug name to search for (e.g., "advill", "tylenol")
        max_entries: Maximum number of candidates to return
    """
    url = f"{BASE_URL}/approximateTerm.json"
    params = {"term": name, "maxEntries": max_entries}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
    candidates = data.get("approximateGroup", {}).get("candidate", [])
    results = []
    for c in candidates:
        results.append({
            "rxcui": c.get("rxcui"),
            "name": c.get("name", ""),
            "score": float(c.get("score", 0.0))
        })
    return results

@mcp.tool()
async def lookup_generic_name(rxcui: str) -> List[str]:
    """
    Lookup the active generic ingredient(s) for a given RxCUI.
    Uses the RxNorm related endpoint with tty=IN+MIN.
    
    Args:
        rxcui: The RxNorm Concept Unique Identifier
    """
    url = f"{BASE_URL}/rxcui/{rxcui}/related.json?tty=IN"
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        
    ingredients = []
    concept_groups = data.get("relatedGroup", {}).get("conceptGroup", [])
    for group in concept_groups:
        if group.get("tty") == "IN":
            props = group.get("conceptProperties", [])
            for prop in props:
                ingredients.append(prop.get("name").lower())
                
    return list(set(ingredients))

@mcp.tool()
async def lookup_brand_name(rxcui: str) -> List[str]:
    """
    Lookup brand names associated with a given RxCUI.
    Uses the RxNorm related endpoint with tty=BN.
    
    Args:
        rxcui: The RxNorm Concept Unique Identifier
    """
    url = f"{BASE_URL}/rxcui/{rxcui}/related.json"
    params = {"tty": "BN"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
    brands = []
    concept_groups = data.get("relatedGroup", {}).get("conceptGroup", [])
    for group in concept_groups:
        if group.get("tty") == "BN":
            props = group.get("conceptProperties", [])
            for prop in props:
                brands.append(prop.get("name"))
                
    return list(set(brands))

@mcp.tool()
async def normalize_drug_name(name: str) -> Dict[str, Any]:
    """
    Normalizes a drug name (brand, generic, misspelled) to its active generic ingredients.
    Returns the top generic match if confidence is high, or multiple candidates if ambiguous.
    
    Args:
        name: The raw drug name to normalize
    """
    # 1. Search for the term
    candidates = await search_drug(name, max_entries=3)
    
    if not candidates:
        return {"status": "not_found", "candidates": []}
        
    # Check the top candidate's score
    top_candidate = candidates[0]
    
    # Let's collect ingredients for the top candidate
    ingredients = await lookup_generic_name(top_candidate["rxcui"])
    
    if not ingredients:
        # If no ingredients found, maybe it IS the ingredient, or just return its name
        cand_name = top_candidate.get("name", "").strip()
        if cand_name:
            ingredients = [cand_name.lower()]
        
    candidate_list = [{"drug": c.get("name", "").lower(), "rxcui": c["rxcui"], "confidence": c["score"]} for c in candidates]
    
    return {
        "status": "success",
        "normalized_drugs": ingredients,
        "candidates": candidate_list,
        "top_score": top_candidate["score"]
    }

if __name__ == "__main__":
    mcp.run(transport="stdio")
