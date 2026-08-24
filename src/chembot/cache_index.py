from __future__ import annotations
import hashlib
import re
from typing import Any, Dict, List, Tuple

SECTIONS_LABEL = [
    "indications_and_usage",
    "contraindications",
    "warnings",
    "warnings_and_cautions",
    "adverse_reactions",
    "drug_interactions",
    "pregnancy",
    "pediatric_use",
    "geriatric_use",
    "dosage_and_administration",
]

def _norm_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        x = "\n".join(str(i) for i in x)
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def chunk_text(text: str, chunk_chars: int = 2200, overlap: int = 250) -> List[str]:
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + chunk_chars)
        out.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return out

def label_record_to_docs(rec: Dict[str, Any], chunk_chars: int, overlap: int) -> List[Tuple[str, Dict[str, Any]]]:
    openfda = rec.get("openfda", {}) or {}
    brand = _norm_text((openfda.get("brand_name") or [""])[0])
    generic = _norm_text((openfda.get("generic_name") or [""])[0])
    substance = _norm_text((openfda.get("substance_name") or [""])[0])
    drug_name = brand or generic or substance or "UNKNOWN_DRUG"

    # Construct a comprehensive drug header so semantic search can find it via brand or generic/substance name
    header_parts = []
    if brand:
        header_parts.append(f"Brand: {brand}")
    if generic:
        header_parts.append(f"Generic: {generic}")
    if substance and substance != generic:
        header_parts.append(f"Substance: {substance}")
    header_str = " | ".join(header_parts) if header_parts else drug_name

    set_id = rec.get("set_id") or ""
    effective_time = rec.get("effective_time") or ""
    base_id = f"label::{set_id}::{effective_time}::{drug_name}"
    base_hash = hashlib.sha256(base_id.encode("utf-8")).hexdigest()[:16]

    docs: List[Tuple[str, Dict[str, Any]]] = []
    for sec in SECTIONS_LABEL:
        text = _norm_text(rec.get(sec))
        if not text:
            continue
        full = f"DRUG: {header_str}\nSECTION: {sec}\nEFFECTIVE_TIME: {effective_time}\n\n{text}"
        for k, ch in enumerate(chunk_text(full, chunk_chars=chunk_chars, overlap=overlap)):
            meta = {
                "source": "openfda_label",
                "drug_name": drug_name,
                "section": sec,
                "effective_time": effective_time,
                "set_id": set_id,
            }
            docs.append((ch, meta | {"chunk_id": f"{base_hash}::{sec}::{k}"}))
    return docs

def event_record_to_docs(rec: Dict[str, Any], chunk_chars: int, overlap: int) -> List[Tuple[str, Dict[str, Any]]]:
    # events are noisy; we store a compact textual summary
    safetyreportid = rec.get("safetyreportid") or ""
    receive_date = rec.get("receivedate") or ""
    base_id = f"event::{safetyreportid}::{receive_date}"
    base_hash = hashlib.sha256(base_id.encode("utf-8")).hexdigest()[:16]

    reactions = []
    for r in rec.get("patient", {}).get("reaction", []) or []:
        rt = r.get("reactionmeddrapt")
        if rt:
            reactions.append(rt)

    drugs = []
    for d in rec.get("patient", {}).get("drug", []) or []:
        mp = d.get("medicinalproduct")
        if mp:
            drugs.append(mp)

    summary = (
        f"ADVERSE EVENT REPORT\n"
        f"safetyreportid={safetyreportid}\nreceivedate={receive_date}\n"
        f"reactions={', '.join(sorted(set(reactions)))}\n"
        f"drugs={', '.join(sorted(set(drugs)))}\n"
    )

    docs: List[Tuple[str, Dict[str, Any]]] = []
    for k, ch in enumerate(chunk_text(summary, chunk_chars=chunk_chars, overlap=overlap)):
        meta = {
            "source": "openfda_event",
            "receivedate": receive_date,
            "safetyreportid": safetyreportid,
        }
        docs.append((ch, meta | {"chunk_id": f"{base_hash}::summary::{k}"}))
    return docs

def pubmed_record_to_docs(rec: Dict[str, Any], chunk_chars: int, overlap: int) -> List[Tuple[str, Dict[str, Any]]]:
    pmid = rec.get("pmid") or ""
    title = _norm_text(rec.get("title"))
    abstract = _norm_text(rec.get("abstract"))
    authors = rec.get("authors") or []
    journal = _norm_text(rec.get("journal"))
    year = rec.get("year") or ""
    pub_types = rec.get("publication_types") or []
    mesh_terms = rec.get("mesh_terms") or []

    if not abstract:
        return []

    base_id = f"pubmed::{pmid}"
    base_hash = hashlib.sha256(base_id.encode("utf-8")).hexdigest()[:16]

    author_str = ", ".join(authors[:5])
    if len(authors) > 5:
        author_str += " et al."

    full = (
        f"PUBMED ARTICLE\n"
        f"PMID={pmid}\n"
        f"TITLE={title}\n"
        f"JOURNAL={journal} ({year})\n"
        f"AUTHORS={author_str}\n\n"
        f"[ABSTRACT]\n{abstract}"
    )

    meta_base = {
        "source": "pubmed",
        "pmid": pmid,
        "title": title,
        "journal": journal,
        "year": year,
        "authors": "; ".join(authors),
        "publication_types": "; ".join(pub_types),
        "mesh_terms": "; ".join(mesh_terms),
    }

    docs: List[Tuple[str, Dict[str, Any]]] = []
    for k, ch in enumerate(chunk_text(full, chunk_chars=chunk_chars, overlap=overlap)):
        docs.append((ch, meta_base | {"chunk_id": f"{base_hash}::abstract::{k}"}))
    return docs