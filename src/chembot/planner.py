import difflib
import re
from typing import List, Literal, Dict, Set
from pydantic import BaseModel, Field
from .rxnorm_mcp_client import RxNormMCPClient


RouteType = Literal["label", "event", "pubmed"]


class QueryPlan(BaseModel):
    normalized_drugs: List[str] = Field(default_factory=list, description="Normalized drug names (e.g., 'ibuprofen')")
    normalized_terms: List[str] = Field(default_factory=list, description="Normalized medical terms")
    drug_mappings: Dict[str, List[str]] = Field(default_factory=dict, description="Mappings from query drug tokens to normalized generic ingredients")
    routes: List[RouteType] = Field(default_factory=list, description="Selected retrieval branches")
    label_queries: List[str] = Field(default_factory=list, description="Optimized openFDA label queries")
    event_queries: List[str] = Field(default_factory=list, description="Optimized openFDA event queries")
    pubmed_queries: List[str] = Field(default_factory=list, description="Optimized PubMed queries")
    explanation: str = Field(default="", description="Explanation of the planning logic")


# Dictionaries for normalization
TERM_SYNONYMS = {
    "adverse effects": ["adverse", "side effect", "side effects", "side affects", "reaction", "reactions", "events", "adverse reaction", "adverse reactions", "faers"],
    "mechanism": ["mechanism", "mode of action", "pathway", "molecular", "biological"],
    "contraindication": ["contraindication", "contraindications", "contraindicatons", "do not use", "avoid"],
    "pregnancy": ["pregnancy", "pregnant", "pregnncy", "maternal", "fetal"],
    "pediatric": ["pediatric", "child", "children", "infant"],
    "geriatric": ["geriatric", "elderly", "older"],
    "dosage": ["dosage", "dose", "administration"],
    "indication": ["indication", "use", "uses", "treats", "treatment"],
    "clinical trial": ["clinical trial", "trial", "rct", "randomized"],
    "systematic review": ["systematic review", "review", "meta-analysis"],
    "compare": ["compare", "comparison", "vs", "versus"],
    "renal": ["renal", "kidney"],
    "hepatic": ["hepatic", "liver"],
    "pancreatitis": ["pancreatitis", "pancreas inflammation"],
    "alzheimer disease": ["alzheimer", "alzheimers", "alzheimer's", "alzheimer disease"],
}

STOP_WORDS = {
    "what", "are", "the", "of", "during", "in", "and", "or", "to", "for", "is", "safe", "does", "can", "i", 
    "take", "with", "recent", "studies", "exist", "on", "a", "an", "do", "how", "why", "when", "about",
    "any", "if", "my", "your", "has", "have", "had", "been", "was", "were", "it", "that", "this", "these"
}


class QueryPlanner:
    def __init__(self):
        self.term_map = {}
        for canonical, aliases in TERM_SYNONYMS.items():
            for alias in aliases:
                self.term_map[alias.lower()] = canonical
                
        # Initialize RxNorm client
        self.rxnorm = RxNormMCPClient()

    def _fuzzy_match(self, word: str, options: List[str], cutoff: float = 0.8) -> str | None:
        """Finds the closest match for a word from a list of options."""
        matches = difflib.get_close_matches(word.lower(), options, n=1, cutoff=cutoff)
        return matches[0] if matches else None

    def _normalize_terms(self, text: str, lookup_map: Dict[str, str], threshold: float = 0.8) -> List[str]:
        """Normalizes terms/phrases in the text against a lookup map using fuzzy matching."""
        # 1. Exact phrase matching (greedy)
        normalized = set()
        text_lower = text.lower()
        
        # We sort keys by length descending to match longest phrases first
        sorted_keys = sorted(lookup_map.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if key in text_lower:
                normalized.add(lookup_map[key])
                # Remove matched phrase to prevent double matching on substrings
                text_lower = text_lower.replace(key, " ")

        # 2. Fuzzy word matching on remaining tokens
        words = re.findall(r'\b\w+\b', text_lower)
        for word in words:
            if len(word) < 4:  # Skip tiny words for fuzzy matching
                continue
            match = self._fuzzy_match(word, list(lookup_map.keys()), cutoff=threshold)
            if match:
                normalized.add(lookup_map[match])

        return list(normalized)
        
    def _extract_and_normalize_drugs(self, user_query: str, normalized_terms: List[str], explanation: List[str]) -> tuple[List[str], Dict[str, List[str]]]:
        """Extracts potential drug names by filtering out stop words and known terms, then queries RxNorm."""
        text_lower = user_query.lower()
        # Remove all known term aliases from the query text to avoid checking them as drugs
        for alias in sorted(self.term_map.keys(), key=len, reverse=True):
            text_lower = text_lower.replace(alias, " ")
                
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', text_lower)
        potential_drug_tokens = []
        for word in words:
            if word not in STOP_WORDS and len(word) >= 3:
                potential_drug_tokens.append(word)
                
        if not potential_drug_tokens:
            return [], {}
            
        # Try to normalize with RxNorm
        normalized_drugs = set()
        drug_mappings = {}
        # We try to join all potential tokens first to handle phrases like "st johns wort"
        # If that fails or has low confidence, we can try individually.
        combined_candidate = " ".join(potential_drug_tokens)
        
        # Call RxNorm MCP
        try:
            res = self.rxnorm.normalize_drug_name(combined_candidate)
            if res.get("status") == "success":
                top_score = res.get("top_score", 0.0)
                if top_score > 10.0:
                    mapped_ingredients = []
                    for d in res.get("normalized_drugs", []):
                        normalized_drugs.add(d)
                        mapped_ingredients.append(d)
                    if mapped_ingredients:
                        drug_mappings[combined_candidate] = mapped_ingredients
                else:
                    # Confidence is low, let's just record the top candidate if it's the only one
                    explanation.append(f"RxNorm confidence was low ({top_score}) for '{combined_candidate}'.")
                    if top_score > 5.0:
                        mapped_ingredients = []
                        for d in res.get("normalized_drugs", []):
                            normalized_drugs.add(d)
                            mapped_ingredients.append(d)
                        if mapped_ingredients:
                            drug_mappings[combined_candidate] = mapped_ingredients
        except Exception as e:
            explanation.append(f"RxNorm normalization failed: {str(e)}")
            
        # If combined failed to yield anything, try individual tokens
        if not normalized_drugs:
            for token in potential_drug_tokens:
                try:
                    res = self.rxnorm.normalize_drug_name(token)
                    if res.get("status") == "success":
                        top_score = res.get("top_score", 0.0)
                        if top_score > 10.0:
                            mapped_ingredients = []
                            for d in res.get("normalized_drugs", []):
                                normalized_drugs.add(d)
                                mapped_ingredients.append(d)
                            if mapped_ingredients:
                                drug_mappings[token] = mapped_ingredients
                        else:
                            explanation.append(f"RxNorm low confidence ({top_score}) for '{token}'.")
                            if top_score > 5.0:
                                mapped_ingredients = []
                                for d in res.get("normalized_drugs", []):
                                    normalized_drugs.add(d)
                                    mapped_ingredients.append(d)
                                if mapped_ingredients:
                                    drug_mappings[token] = mapped_ingredients
                except Exception:
                    continue

        # Fallback to using raw keywords if synonyms/normalizations are not available
        resolved_keys = set(drug_mappings.keys())
        if combined_candidate not in resolved_keys:
            for token in potential_drug_tokens:
                if token not in resolved_keys:
                    normalized_drugs.add(token)

        # Clean up empty strings or None from normalized_drugs
        normalized_drugs = {d for d in normalized_drugs if d and d.strip()}

        return list(normalized_drugs), drug_mappings

    def plan(self, user_query: str) -> QueryPlan:
        explanation = []
        
        # 1. Normalize terms
        normalized_terms = self._normalize_terms(user_query, self.term_map, threshold=0.85)
        if normalized_terms:
            explanation.append(f"Identified medical terms: {', '.join(normalized_terms)}.")
            
        # 2. Normalize drugs using RxNorm
        normalized_drugs, drug_mappings = self._extract_and_normalize_drugs(user_query, normalized_terms, explanation)
        if normalized_drugs:
            explanation.append(f"Identified drugs (via RxNorm): {', '.join(normalized_drugs)}.")
        else:
            explanation.append("No specific drugs identified.")

        # 3. Determine routes
        routes = set()
        
        # Route: FAERS (Events)
        if "adverse effects" in normalized_terms or "pancreatitis" in normalized_terms:
            routes.add("event")
            
        # Route: PubMed
        pubmed_triggers = {"mechanism", "clinical trial", "systematic review", "compare", "pregnancy", "pediatric", "geriatric", "renal", "hepatic", "pancreatitis", "alzheimer disease"}
        if any(term in pubmed_triggers for term in normalized_terms):
            routes.add("pubmed")
            
        # Route: FDA Label
        label_triggers = {"contraindication", "pregnancy", "pediatric", "geriatric", "dosage", "indication", "renal", "hepatic", "compare"}
        if any(term in label_triggers for term in normalized_terms):
            routes.add("label")
            
        # Default fallback route if no specific terms triggered anything
        if not routes:
            routes.add("label")
            explanation.append("Defaulting to 'label' route as no specific intent was detected.")
            
        explanation.append(f"Selected routes: {', '.join(routes)}.")

        # 4. Generate Queries
        label_queries = []
        event_queries = []
        pubmed_queries = []

        drug_str = " ".join(normalized_drugs) if normalized_drugs else ""
        term_str = " ".join(normalized_terms) if normalized_terms else ""

        if "label" in routes:
            for drug in normalized_drugs:
                label_queries.append(f'openfda.generic_name:"{drug}" OR openfda.brand_name:"{drug}" OR openfda.substance_name:"{drug}"')
            if not label_queries:
                # Fallback to direct raw query if no drugs found
                cleaned_q = user_query.replace('?', '').replace('"', '')
                label_queries.append(f'openfda.brand_name:"{cleaned_q}" OR openfda.generic_name:"{cleaned_q}" OR openfda.substance_name:"{cleaned_q}"')
                
        if "event" in routes:
            if normalized_drugs and normalized_terms:
                for drug in normalized_drugs:
                    for term in normalized_terms:
                        event_queries.append(f'medicinalproduct:"{drug.upper()}" AND reactionmeddrapt:"{term}"')
            elif normalized_drugs:
                for drug in normalized_drugs:
                    event_queries.append(f'medicinalproduct:"{drug.upper()}"')
            else:
                # Fallback
                cleaned_q = user_query.replace('?', '').replace('"', '')
                event_queries.append(f'reactionmeddrapt:"{cleaned_q}"')
                
        if "pubmed" in routes:
            # Combine drugs and terms for pubmed
            if drug_str or term_str:
                pubmed_queries.append(f"{drug_str} {term_str}".strip())
            else:
                cleaned_q = user_query.replace('?', '').replace('"', '')
                pubmed_queries.append(cleaned_q.strip())

        return QueryPlan(
            normalized_drugs=normalized_drugs,
            normalized_terms=normalized_terms,
            drug_mappings=drug_mappings,
            routes=list(routes),
            label_queries=label_queries,
            event_queries=event_queries,
            pubmed_queries=pubmed_queries,
            explanation=" ".join(explanation)
        )
