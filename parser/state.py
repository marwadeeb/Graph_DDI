"""
state.py — shared mutable state passed to every parser module.
Holds auto-increment counters and deduplication look-up tables.
"""


class ParserState:
    def __init__(self):
        # ── References ────────────────────────────────────────────────────────
        # Maps dedup key -> ref_pk (int)
        # key = ("article", pubmed_id_or_citation) | ("textbook", ...) | ...
        self.refs_seen: dict = {}
        self.ref_counter: int = 0

        # ── Categories ────────────────────────────────────────────────────────
        # Maps (category_name_lower, mesh_id) -> category_id (int)
        self.cats_seen: dict = {}
        self.cat_counter: int = 0

        # ── Pathways ──────────────────────────────────────────────────────────
        # Set of smpdb_ids already written to pathways.csv
        self.pathways_seen: set = set()

        # ── Polypeptides ──────────────────────────────────────────────────────
        # Set of polypeptide_ids (UniProt) already written to polypeptides.csv
        self.polypeptides_seen: set = set()

        # ── Interactants ──────────────────────────────────────────────────────
        # Set of interactant_ids (BE-IDs) already written to interactants.csv
        self.interactants_seen: set = set()

        # ── Reactions ─────────────────────────────────────────────────────────
        self.reaction_counter: int = 0

        # ── Products ──────────────────────────────────────────────────────────
        self.product_counter: int = 0

        # ── Drug counter (for progress reporting) ─────────────────────────────
        self.drug_count: int = 0
