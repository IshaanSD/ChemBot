import os
import requests
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("pubmed-server")

TIMEOUT = 20
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def _ncbi_params() -> Dict[str, str]:
    """Return common NCBI params (email + optional api_key)."""
    params: Dict[str, str] = {}
    email = os.environ.get("NCBI_EMAIL")
    if email:
        params["email"] = email
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def _esearch(query: str, max_results: int = 5) -> List[str]:
    """Search PubMed and return a list of PMIDs."""
    params = _ncbi_params() | {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "xml",
    }
    try:
        resp = requests.get(ESEARCH_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return []

    root = ET.fromstring(resp.content)
    return [id_el.text for id_el in root.findall(".//IdList/Id") if id_el.text]


def _parse_article(article_el: ET.Element) -> Dict[str, Any]:
    """Parse a single <PubmedArticle> XML element into a dict."""
    citation = article_el.find("MedlineCitation")
    if citation is None:
        return {}

    # PMID
    pmid_el = citation.find("PMID")
    pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""

    article = citation.find("Article")
    if article is None:
        return {"pmid": pmid}

    # Title
    title_el = article.find("ArticleTitle")
    title = title_el.text if title_el is not None and title_el.text else ""

    # Abstract — may have multiple AbstractText elements (structured abstract)
    abstract_parts: List[str] = []
    abstract_el = article.find("Abstract")
    if abstract_el is not None:
        for at in abstract_el.findall("AbstractText"):
            label = at.get("Label", "")
            text = "".join(at.itertext()).strip()
            if text:
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
    abstract = "\n".join(abstract_parts)

    # Authors
    authors: List[str] = []
    author_list = article.find("AuthorList")
    if author_list is not None:
        for author_el in author_list.findall("Author"):
            last = author_el.find("LastName")
            initials = author_el.find("Initials")
            if last is not None and last.text:
                name = last.text
                if initials is not None and initials.text:
                    name += f" {initials.text}"
                authors.append(name)

    # Journal + Year
    journal_el = article.find("Journal/Title")
    journal = journal_el.text if journal_el is not None and journal_el.text else ""

    year = ""
    # Try Article date first, then PubDate
    for date_path in [
        "Journal/JournalIssue/PubDate/Year",
        "ArticleDate/Year",
    ]:
        year_el = article.find(date_path)
        if year_el is not None and year_el.text:
            year = year_el.text
            break
    # Fallback to MedlineDate if Year element is missing
    if not year:
        medline_date_el = article.find("Journal/JournalIssue/PubDate/MedlineDate")
        if medline_date_el is not None and medline_date_el.text:
            year = medline_date_el.text[:4]

    # Publication Types
    pub_types: List[str] = []
    for pt in article.findall("PublicationTypeList/PublicationType"):
        if pt.text:
            pub_types.append(pt.text)

    # MeSH Terms
    mesh_terms: List[str] = []
    mesh_list = citation.find("MeshHeadingList")
    if mesh_list is not None:
        for mh in mesh_list.findall("MeshHeading/DescriptorName"):
            if mh.text:
                mesh_terms.append(mh.text)

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal,
        "year": year,
        "publication_types": pub_types,
        "mesh_terms": mesh_terms,
    }


def _efetch(pmids: List[str]) -> List[Dict[str, Any]]:
    """Fetch full PubMed article records by PMID list."""
    if not pmids:
        return []

    params = _ncbi_params() | {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    try:
        resp = requests.get(EFETCH_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException:
        return []

    root = ET.fromstring(resp.content)
    articles = []
    for article_el in root.findall("PubmedArticle"):
        parsed = _parse_article(article_el)
        if parsed and parsed.get("pmid"):
            articles.append(parsed)
    return articles


# ─── MCP Tools ──────────────────────────────────────────────

@mcp.tool()
def search_pubmed(query: str, max_results: int = 5) -> dict:
    """Search PubMed for biomedical literature matching a query.

    Args:
        query: PubMed search query (e.g., 'ibuprofen COX inhibition mechanism').
        max_results: Maximum number of articles to return (default 5).
    """
    pmids = _esearch(query, max_results)
    articles = _efetch(pmids)
    return {"results": articles}


@mcp.tool()
def search_clinical_trials(query: str, max_results: int = 5) -> dict:
    """Search PubMed for clinical trial publications matching a query.

    Args:
        query: PubMed search query (e.g., 'metformin PCOS').
        max_results: Maximum number of articles to return (default 5).
    """
    full_query = f'{query} AND "Clinical Trial"[Publication Type]'
    pmids = _esearch(full_query, max_results)
    articles = _efetch(pmids)
    return {"results": articles}


@mcp.tool()
def search_systematic_reviews(query: str, max_results: int = 5) -> dict:
    """Search PubMed for systematic review publications matching a query.

    Args:
        query: PubMed search query (e.g., 'melatonin insomnia').
        max_results: Maximum number of articles to return (default 5).
    """
    full_query = f'{query} AND "Systematic Review"[Publication Type]'
    pmids = _esearch(full_query, max_results)
    articles = _efetch(pmids)
    return {"results": articles}


@mcp.tool()
def search_meta_analyses(query: str, max_results: int = 5) -> dict:
    """Search PubMed for meta-analysis publications matching a query.

    Args:
        query: PubMed search query (e.g., 'statins cardiovascular outcomes').
        max_results: Maximum number of articles to return (default 5).
    """
    full_query = f'{query} AND "Meta-Analysis"[Publication Type]'
    pmids = _esearch(full_query, max_results)
    articles = _efetch(pmids)
    return {"results": articles}


@mcp.tool()
def fetch_abstracts(pmids: list[str]) -> dict:
    """Fetch PubMed article details by a list of PMIDs.

    Args:
        pmids: List of PubMed IDs to fetch (e.g., ['12345678', '87654321']).
    """
    articles = _efetch(pmids)
    return {"results": articles}


if __name__ == "__main__":
    mcp.run(transport="stdio")
