"""
main_parser.py — single-pass streaming XML parser for DrugBank full database.

Streams through the 45M-line XML file ONE time using lxml.etree.iterparse,
processes one <drug> element at a time, and writes all 27 CSV files.
Memory usage stays flat regardless of file size.

Usage:
    python main_parser.py
"""
import os
import sys
import time
from lxml import etree

from config import XML_PATH, OUTPUT_DIR, NP, PROGRESS_EVERY, SCHEMA
from state import ParserState
from utils import open_writer, write_rows, get_primary_id

import parse_core
import parse_references
import parse_commercial
import parse_pharmacological
import parse_interactions
import parse_pathways
import parse_proteins


# Tables written by multiple modules (refs + external_identifiers + ref_associations)
# are handled by merging results before writing — same writer used for all.

EXTRACTORS = [
    parse_core.extract,
    parse_references.extract,
    parse_commercial.extract,
    parse_pharmacological.extract,
    parse_interactions.extract,
    parse_pathways.extract,
    parse_proteins.extract,
]


def main():
    print(f"[main_parser] Output directory: {OUTPUT_DIR}")
    print(f"[main_parser] Parsing: {XML_PATH}")
    print(f"[main_parser] Opening {len(SCHEMA)} CSV writers …")

    # Open all CSV writers
    handles = {}
    writers = {}
    for table in SCHEMA:
        f, w = open_writer(table)
        handles[table] = f
        writers[table] = w

    state = ParserState()
    t0 = time.time()

    try:
        # iterparse: fire "end" event for every completed <drug> element
        context = etree.iterparse(
            XML_PATH,
            events=("end",),
            tag=f"{NP}drug",
            recover=True,
        )

        for event, drug_el in context:
            # Only process TOP-LEVEL <drug> elements (direct children of <drugbank>).
            # Nested <drug> elements appear inside <pathways>/<pathway>/<drugs>
            # and have a very different, minimal structure — skipping them here
            # prevents duplicate/sparse rows; they are captured by parse_pathways.py.
            parent = drug_el.getparent()
            if parent is None or parent.tag != f"{NP}drugbank":
                continue   # nested drug — do NOT clear (still needed by parent)

            primary_id = get_primary_id(drug_el)
            if not primary_id:
                drug_el.clear()
                continue

            state.drug_count += 1

            # Run every extractor and write returned rows immediately
            for extractor in EXTRACTORS:
                result = extractor(drug_el, primary_id, state)
                for table_name, rows in result.items():
                    if rows:
                        write_rows(writers[table_name], rows)

            # Free memory: clear the processed element and its preceding siblings
            drug_el.clear()
            parent = drug_el.getparent()
            if parent is not None:
                while parent[0] is not drug_el:
                    del parent[0]

            # Progress report
            if state.drug_count % PROGRESS_EVERY == 0:
                elapsed = time.time() - t0
                rate = state.drug_count / elapsed
                print(f"  [{state.drug_count:>6} drugs | "
                      f"{elapsed:6.1f}s | {rate:.0f} drugs/s] "
                      f"refs={state.ref_counter} "
                      f"cats={state.cat_counter} "
                      f"polypeptides={len(state.polypeptides_seen)}")

        elapsed = time.time() - t0
        print(f"\n[main_parser] Done in {elapsed:.1f}s")
        print(f"  Drugs processed   : {state.drug_count:,}")
        print(f"  Unique references : {state.ref_counter:,}")
        print(f"  Unique categories : {state.cat_counter:,}")
        print(f"  Unique pathways   : {len(state.pathways_seen):,}")
        print(f"  Unique polypeptides: {len(state.polypeptides_seen):,}")
        print(f"  Unique interactants: {len(state.interactants_seen):,}")
        print(f"  Reactions         : {state.reaction_counter:,}")
        print(f"  Products          : {state.product_counter:,}")

    finally:
        for f in handles.values():
            f.close()

    # Verify all 27 CSV files were created
    print("\n[main_parser] CSV files written:")
    total_bytes = 0
    for table in SCHEMA:
        path = os.path.join(OUTPUT_DIR, f"{table}.csv")
        size = os.path.getsize(path) if os.path.exists(path) else 0
        total_bytes += size
        print(f"  {table}.csv  —  {size:,} bytes")
    print(f"\n  Total output: {total_bytes / 1024 / 1024:.1f} MB")

    return state


if __name__ == "__main__":
    main()
