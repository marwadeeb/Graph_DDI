"""
utils.py — shared helper functions for all parser modules.
"""
import csv
import os
from config import NP, OUTPUT_DIR, SCHEMA


# ── Text helpers ─────────────────────────────────────────────────────────────

def clean(text):
    """Strip whitespace; return None for empty/None strings."""
    if text is None:
        return None
    s = str(text).strip()
    return s if s else None


def t(el, tag):
    """Get cleaned text of a direct child element (namespace-aware)."""
    if el is None:
        return None
    child = el.find(f"{NP}{tag}")
    if child is None:
        return None
    return clean(child.text)


def a(el, attr, default=None):
    """Get cleaned attribute value from element."""
    v = el.get(attr, default)
    return clean(v) if v is not None else default


def get_primary_id(drug_el):
    """Return the primary DrugBank ID for a <drug> element."""
    for id_el in drug_el.findall(f"{NP}drugbank-id"):
        if id_el.get("primary") == "true":
            return clean(id_el.text)
    # Fallback: first id if primary flag absent
    ids = drug_el.findall(f"{NP}drugbank-id")
    return clean(ids[0].text) if ids else None


# ── CSV helpers ───────────────────────────────────────────────────────────────

def open_writer(table_name):
    """Open a CSV writer for the given table; writes header row."""
    path = os.path.join(OUTPUT_DIR, f"{table_name}.csv")
    f = open(path, "w", newline="", encoding="utf-8")
    cols = SCHEMA[table_name]
    writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore",
                            quoting=csv.QUOTE_ALL)
    writer.writeheader()
    return f, writer


def write_rows(writer, rows):
    """Write a list of row-dicts to a DictWriter (ignores unknown keys)."""
    for row in rows:
        writer.writerow(row)


# ── Reference extraction (shared by parse_references + parse_proteins) ───────

def extract_ref_list(refs_el, state):
    """
    Parse a <references> element (reference-list-type) and return:
      new_ref_rows : list[dict]  — rows for the 'references' table (newly seen)
      ref_pks      : list[int]   — ref_pk for EVERY reference found (for associations)

    Deduplication keys:
      article    → pubmed_id  (or citation truncated to 300 chars if no pubmed_id)
      textbook   → isbn + "|" + citation[:200]
      link       → url
      attachment → title + "|" + url
    """
    new_refs = []
    ref_pks = []

    if refs_el is None:
        return new_refs, ref_pks

    # ── Articles ──────────────────────────────────────────────────────────────
    articles_el = refs_el.find(f"{NP}articles")
    if articles_el is not None:
        for art in articles_el.findall(f"{NP}article"):
            pubmed_id = t(art, "pubmed-id")
            citation  = t(art, "citation")
            ref_id    = t(art, "ref-id")
            if pubmed_id:
                key = ("article", pubmed_id)
            elif citation:
                key = ("article", citation[:300])
            else:
                continue
            if key not in state.refs_seen:
                state.ref_counter += 1
                rpk = state.ref_counter
                state.refs_seen[key] = rpk
                new_refs.append({
                    "ref_pk": rpk, "ref_type": "article", "ref_id": ref_id,
                    "pubmed_id": pubmed_id, "isbn": None,
                    "title": None, "url": None, "citation": citation,
                })
            ref_pks.append(state.refs_seen[key])

    # ── Textbooks ─────────────────────────────────────────────────────────────
    textbooks_el = refs_el.find(f"{NP}textbooks")
    if textbooks_el is not None:
        for tb in textbooks_el.findall(f"{NP}textbook"):
            isbn     = t(tb, "isbn")
            citation = t(tb, "citation")
            ref_id   = t(tb, "ref-id")
            key_str  = (isbn or "") + "|" + (citation[:200] if citation else "")
            key = ("textbook", key_str)
            if not key_str.strip("|"):
                continue
            if key not in state.refs_seen:
                state.ref_counter += 1
                rpk = state.ref_counter
                state.refs_seen[key] = rpk
                new_refs.append({
                    "ref_pk": rpk, "ref_type": "textbook", "ref_id": ref_id,
                    "pubmed_id": None, "isbn": isbn,
                    "title": None, "url": None, "citation": citation,
                })
            ref_pks.append(state.refs_seen[key])

    # ── Links ─────────────────────────────────────────────────────────────────
    links_el = refs_el.find(f"{NP}links")
    if links_el is not None:
        for lnk in links_el.findall(f"{NP}link"):
            url    = t(lnk, "url")
            title  = t(lnk, "title")
            ref_id = t(lnk, "ref-id")
            if not url:
                continue
            key = ("link", url)
            if key not in state.refs_seen:
                state.ref_counter += 1
                rpk = state.ref_counter
                state.refs_seen[key] = rpk
                new_refs.append({
                    "ref_pk": rpk, "ref_type": "link", "ref_id": ref_id,
                    "pubmed_id": None, "isbn": None,
                    "title": title, "url": url, "citation": None,
                })
            ref_pks.append(state.refs_seen[key])

    # ── Attachments ───────────────────────────────────────────────────────────
    attachments_el = refs_el.find(f"{NP}attachments")
    if attachments_el is not None:
        for att in attachments_el.findall(f"{NP}attachment"):
            url    = t(att, "url")
            title  = t(att, "title")
            ref_id = t(att, "ref-id")
            key_str = (title or "") + "|" + (url or "")
            key = ("attachment", key_str)
            if not key_str.strip("|"):
                continue
            if key not in state.refs_seen:
                state.ref_counter += 1
                rpk = state.ref_counter
                state.refs_seen[key] = rpk
                new_refs.append({
                    "ref_pk": rpk, "ref_type": "attachment", "ref_id": ref_id,
                    "pubmed_id": None, "isbn": None,
                    "title": title, "url": url, "citation": None,
                })
            ref_pks.append(state.refs_seen[key])

    return new_refs, ref_pks
