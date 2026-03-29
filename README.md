# DrugBank XML → CSV Normalization Pipeline

**Source:** DrugBank Full Database v5.1 (exported 2026-03-05)
**Input:** `full database.xml` (~45.9 M lines, ~19,842 drugs)
**Output:** 27 normalized CSV files in `csv_output/`

---

## Pipeline Overview

```
Step 1 — Parse & Validate          csv_output/          (27 CSVs, 707 MB, all 19,842 drugs)
Step 2 — Dedup Interactions        step2_output/        (1 CSV, 161 MB, undirected DDI pairs)
Step 3 — FDA-Approved Subset       step3_output/        (27 CSVs, 345 MB, 4,795 drugs)
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Step 1: parse full database + validate
python parser/run_all.py                    # parse + validate (one command)
python parser/main_parser.py               # parse only
python parser/validate.py                  # validate only (after parse)

# Step 2: deduplicate DDI pairs (A↔B = B↔A)
python pipeline/step2_dedup_interactions.py

# Step 3: filter to FDA-approved drugs only (run after step 2)
python pipeline/step3_fda_approved.py
```

**Python 3.9+ required.**

---

## File Structure

```
├── README.md
├── .gitignore
├── requirements.txt
│
├── parser/                          Step 1 — XML parsing modules
│   ├── config.py                    shared paths, constants, full schema definition
│   ├── state.py                     ParserState class (dedup counters & lookup tables)
│   ├── utils.py                     helper functions (XML helpers, CSV writers, ref dedup)
│   ├── parse_core.py                → drugs, drug_ids, drug_attributes, drug_properties, external_identifiers
│   ├── parse_references.py          → references, reference_associations (drug general refs)
│   ├── parse_commercial.py          → salts, products, drug_commercial_entities, mixtures, prices
│   ├── parse_pharmacological.py     → categories, drug_categories, dosages, atc_codes, patents
│   ├── parse_interactions.py        → drug_interactions, drug_snp_data
│   ├── parse_pathways.py            → pathways, pathway_members, reactions
│   ├── parse_proteins.py            → interactants, drug_interactants, polypeptides,
│   │                                   interactant_polypeptides, polypeptide_attributes
│   │                                   + appends to: references, reference_associations,
│   │                                                 external_identifiers
│   ├── main_parser.py               streaming XML loop — calls all parsers, writes CSVs
│   ├── validate.py                  16-check validation suite
│   └── run_all.py                   entry point (parser → validator)
│
├── pipeline/                        Steps 2 & 3 — post-processing
│   ├── step2_dedup_interactions.py  deduplicates DDI pairs to undirected edges
│   └── step3_fda_approved.py        filters all tables to FDA-approved drugs only
│
└── data/
    ├── step1_full/                  27 CSVs, 707 MB — full database  [gitignored]
    ├── step2_dedup/                 1 CSV,  161 MB — undirected DDI  [gitignored]
    └── step3_approved/              27 CSVs, 345 MB — FDA-approved   [tracked on GitHub]
```

---

## Tables (27)

### 1. `drugs`
*One row per drug. All scalar fields plus inlined ClassyFire classification.*

| Column | Type | Description |
|---|---|---|
| drugbank_id | PK | Primary DrugBank ID (e.g. DB00001) |
| name | text | Drug name |
| drug_type | text | `small molecule` or `biotech` |
| description | text | Full drug description |
| cas_number | text | CAS Registry Number |
| unii | text | FDA Unique Ingredient Identifier |
| average_mass | float | Average molecular mass |
| monoisotopic_mass | float | Monoisotopic molecular mass |
| state | text | Physical state: `solid` / `liquid` / `gas` |
| indication | text | Therapeutic indications |
| pharmacodynamics | text | Pharmacodynamics description |
| mechanism_of_action | text | Mechanism of action |
| toxicity | text | Toxicity and overdose information |
| metabolism | text | Metabolic pathway description |
| absorption | text | Absorption characteristics |
| half_life | text | Plasma half-life |
| protein_binding | text | Protein binding % or description |
| route_of_elimination | text | Elimination route |
| volume_of_distribution | text | Vd description |
| clearance | text | Clearance description |
| synthesis_reference | text | Synthesis reference text |
| fda_label_url | text | URL to FDA label (when available) |
| msds_url | text | URL to MSDS (when available) |
| classification_description | text | ClassyFire description |
| classification_direct_parent | text | ClassyFire direct parent class |
| classification_kingdom | text | ClassyFire kingdom |
| classification_superclass | text | ClassyFire superclass |
| classification_class | text | ClassyFire class |
| classification_subclass | text | ClassyFire subclass |
| created_date | date | Record creation date |
| updated_date | date | Record last-updated date |

---

### 2. `drug_ids`
*All DrugBank IDs per drug — primary + all secondary/legacy formats.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| legacy_id | The ID value (DB#####, BIOD#####, BTD#####, APRD#####, EXPT#####, NUTR#####) |
| is_primary | `True` for the canonical ID used as PK in `drugs` |

---

### 3. `drug_attributes`
*Catch-all table for 9 multi-valued list fields. Use `attr_type` to filter.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| attr_type | See values below |
| value | Primary value |
| value2 | Secondary value (type-dependent) |
| value3 | Tertiary value (type-dependent) |

**`attr_type` values:**

| attr_type | value | value2 | value3 |
|---|---|---|---|
| `group` | approved / illicit / experimental / withdrawn / nutraceutical / investigational / vet_approved | — | — |
| `synonym` | synonym text | language code | coder |
| `affected_organism` | organism name | — | — |
| `food_interaction` | interaction description text | — | — |
| `sequence` | FASTA sequence string | format (FASTA) | — |
| `ahfs_code` | AHFS pharmacologic-therapeutic classification code | — | — |
| `pdb_entry` | PDB structure ID | — | — |
| `classification_alt_parent` | Alternative parent class name (ClassyFire) | — | — |
| `classification_substituent` | Substituent name (ClassyFire) | — | — |

---

### 4. `drug_properties`
*Calculated and experimental physicochemical properties.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| property_class | `calculated` (ChemAxon/ALOGPS) or `experimental` |
| kind | Property type (logP, SMILES, Melting Point, Water Solubility, IUPAC Name, etc.) |
| value | Property value |
| source | Source tool/database (ChemAxon, ALOGPS, etc.) |

---

### 5. `external_identifiers`
*Cross-database IDs for drugs, polypeptides, salts; plus drug external web links.*

| Column | Description |
|---|---|
| entity_type | `drug` / `polypeptide` / `salt` / `drug_link` |
| entity_id | Primary ID of the entity (drugbank_id, UniProt ID, or salt_id) |
| resource | Database or resource name |
| identifier | The ID value (or URL for `drug_link`) |

**Resources for `drug`:** UniProtKB, Wikipedia, ChEBI, ChEMBL, PubChem Compound, PubChem Substance, Drugs Product Database (DPD), KEGG Compound, KEGG Drug, ChemSpider, BindingDB, National Drug Code Directory, GenBank, Therapeutic Targets Database, PharmGKB, PDB, IUPHAR, Guide to Pharmacology, ZINC, RxCUI
**Resources for `drug_link`:** RxList, PDRhealth, Drugs.com
**Resources for `polypeptide`:** UniProtKB, UniProt Accession, HUGO Gene Nomenclature Committee (HGNC), Human Protein Reference Database (HPRD), GenAtlas, GeneCards, GenBank Gene Database, GenBank Protein Database, ChEMBL, IUPHAR, Guide to Pharmacology
**Resources for `salt`:** DrugBank (primary), DrugBank (legacy)

---

### 6. `references`
*Globally deduplicated bibliography — all reference types in one table.*

| Column | Description |
|---|---|
| ref_pk | Auto-increment PK (integer) |
| ref_type | `article` / `textbook` / `link` / `attachment` |
| ref_id | DrugBank internal reference ID (e.g. A1, L41539) — first-seen value |
| pubmed_id | PubMed ID (articles only) |
| isbn | ISBN (textbooks only) |
| title | Link or attachment title |
| url | URL (links and attachments) |
| citation | Full citation text (articles and textbooks) |

**Dedup keys:** articles → pubmed_id; textbooks → isbn+citation; links → url; attachments → title+url

---

### 7. `reference_associations`
*Links each reference to its usage context.*

| Column | Description |
|---|---|
| ref_pk | FK → references |
| drugbank_id | FK → drugs (the drug whose entry contains this reference) |
| interactant_id | FK → interactants — NULL means drug general reference; BE-ID means reference cited within that interactant's entry for this drug |

---

### 8. `salts`
*Salt forms of drugs (DBSALT IDs).*

| Column | Description |
|---|---|
| salt_id | Primary salt ID (DBSALT######) |
| drugbank_id | FK → drugs (parent drug) |
| name | Salt name |
| unii | FDA Unique Ingredient Identifier for the salt |
| cas_number | CAS number |
| inchikey | InChIKey |
| average_mass | Average molecular mass |
| monoisotopic_mass | Monoisotopic molecular mass |

*Legacy salt IDs → `external_identifiers` with entity_type='salt'*

---

### 9. `products`
*Marketed drug products from FDA NDC, Health Canada DPD, and EMA.*

| Column | Description |
|---|---|
| product_id | Auto-increment PK |
| drugbank_id | FK → drugs |
| name | Product/brand name |
| labeller | Marketing authorization holder |
| ndc_id | FDA NDC ID |
| ndc_product_code | FDA NDC product code |
| dpd_id | Health Canada DPD ID |
| ema_product_code | EMA product code |
| ema_ma_number | EMA marketing authorisation number |
| started_marketing_on | Marketing start date |
| ended_marketing_on | Marketing end date |
| dosage_form | Dosage form (tablet, injection, etc.) |
| strength | Strength (e.g. "20 mg") |
| route | Route of administration |
| fda_application_number | FDA NDA/ANDA number |
| generic | Boolean — generic product |
| over_the_counter | Boolean — OTC product |
| approved | Boolean — currently approved |
| country | `US` / `Canada` / `EU` |
| source | `FDA NDC` / `DPD` / `EMA` |

---

### 10. `drug_commercial_entities`
*Packagers, manufacturers, and international brand names — merged with entity_type.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| entity_type | `packager` / `manufacturer` / `brand` |
| name | Entity name |
| url | Website URL (packagers and manufacturers) |
| company | Company name (brands only) |
| generic_flag | Boolean string — manufacturer produces generics (manufacturers only) |

---

### 11. `mixtures`
*Drug mixture and combination products.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| name | Mixture/product name |
| ingredients | Pipe or comma-delimited ingredient list |
| supplemental_ingredients | Additional ingredients (nullable) |

---

### 12. `prices`
*Drug pricing data (US/Canadian currency).*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| description | Product description (e.g. "Refludan 50 mg vial") |
| cost | Numeric cost value |
| currency | `USD` or `CAD` |
| unit | Unit (e.g. "vial", "tablet") |

---

### 13. `categories`
*Normalized MeSH pharmacological category entities.*

| Column | Description |
|---|---|
| category_id | Auto-increment PK |
| category_name | Category name (e.g. "Anticoagulants") |
| mesh_id | MeSH descriptor ID (e.g. D000925) — may be empty |

---

### 14. `drug_categories`
*Drug ↔ MeSH category many-to-many junction.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| category_id | FK → categories |

---

### 15. `dosages`
*Approved dosage forms, routes, and strengths.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| form | Dosage form (Tablet, Injection, Powder, etc.) |
| route | Route of administration (Oral, Intravenous, etc.) |
| strength | Strength value (e.g. "50 mg") |

---

### 16. `atc_codes`
*WHO ATC classification with full 4-level hierarchy per code.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| atc_code | Full ATC code (e.g. B01AE02) |
| l1_code / l1_name | Most specific subgroup (e.g. B01AE / "Direct thrombin inhibitors") |
| l2_code / l2_name | Pharmacological subgroup |
| l3_code / l3_name | Pharmacological/therapeutic group |
| l4_code / l4_name | Anatomical main group (e.g. B / "BLOOD AND BLOOD FORMING ORGANS") |

---

### 17. `patents`
*Drug patent records.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| number | Patent number |
| country | Country of patent |
| approved_date | Patent approval date |
| expires_date | Patent expiry date |
| pediatric_extension | Boolean — pediatric exclusivity extension |

---

### 18. `drug_interactions`
*Drug–drug interaction (DDI) records — **the core edge table for the GNN**.*

| Column | Description |
|---|---|
| drugbank_id | Source drug (FK → drugs) |
| interacting_drugbank_id | Target drug (FK → drugs) |
| description | Interaction description text |

> **Note:** Both directions are stored (A→B and B→A) as each has a distinct description. The graph builder should treat these as undirected edges. ~2.9 M rows = ~1.45 M unique drug pairs.

---

### 19. `drug_snp_data`
*SNP pharmacogenomics — effects and adverse drug reactions merged.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| snp_type | `effect` or `adverse_reaction` |
| protein_name | Protein affected by SNP |
| gene_symbol | Gene symbol |
| uniprot_id | UniProt ID of the protein |
| rs_id | dbSNP rs ID |
| allele | Variant allele |
| defining_change | Amino acid change (effects only) |
| adverse_reaction | Adverse reaction name (adverse_reactions only) |
| description | Clinical description |
| pubmed_id | Supporting PubMed ID |

---

### 20. `pathways`
*Deduplicated SMPDB pathway entities.*

| Column | Description |
|---|---|
| smpdb_id | PK — SMPDB pathway ID (e.g. SMP0000001) |
| name | Pathway name |
| category | Pathway category (e.g. Drug metabolism, Signaling) |

---

### 21. `pathway_members`
*Drugs and enzymes (UniProt) participating in each pathway.*

| Column | Description |
|---|---|
| smpdb_id | FK → pathways |
| member_type | `drug` or `enzyme` |
| member_id | DrugBank ID (drug) or UniProt ID (enzyme) |
| member_name | Drug name (for drugs); NULL for enzymes |

---

### 22. `reactions`
*Metabolic reaction records. Catalyzing enzymes are serialized in `enzymes`.*

| Column | Description |
|---|---|
| reaction_id | Auto-increment PK |
| drugbank_id | FK → drugs (drug this reaction was listed under) |
| sequence | Reaction step identifier |
| left_element_id | Substrate DrugBank ID |
| left_element_name | Substrate name |
| right_element_id | Product DrugBank ID |
| right_element_name | Product name |
| enzymes | Pipe-delimited enzyme triples: `db_id\|name\|uniprot_id\|\|db_id\|name\|uniprot_id` |

---

### 23. `interactants`
*Binding entity records — identified by DrugBank BE-IDs. Deduplicated globally.*

| Column | Description |
|---|---|
| interactant_id | PK — BE-ID (e.g. BE0000048) |
| name | Protein/entity name (e.g. Prothrombin) |
| organism | Source organism |

---

### 24. `drug_interactants`
*Drug ↔ binding entity junction — the drug–protein interaction record.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| interactant_id | FK → interactants |
| role | `target` / `enzyme` / `carrier` / `transporter` |
| position | Ordering position within the role group |
| known_action | `yes` / `no` / `unknown` |
| actions | Pipe-delimited pharmacological actions (e.g. `inhibitor\|substrate`) |
| inhibition_strength | Enzyme inhibition strength (enzymes only) |
| induction_strength | Enzyme induction strength (enzymes only) |

---

### 25. `polypeptides`
*UniProt protein records. Deduplicated globally by UniProt ID.*

| Column | Description |
|---|---|
| polypeptide_id | PK — UniProt ID (e.g. P00734) |
| source | `Swiss-Prot` or `TrEMBL` |
| name | Protein name |
| general_function | General functional description |
| specific_function | Specific function (GO molecular function) |
| gene_name | Gene symbol (HGNC) |
| locus | Chromosomal locus |
| cellular_location | Subcellular location |
| transmembrane_regions | Transmembrane region positions |
| signal_regions | Signal peptide positions |
| theoretical_pi | Theoretical isoelectric point |
| molecular_weight | Molecular weight (Da) |
| chromosome_location | Chromosome number |
| organism | Source organism name |
| ncbi_taxonomy_id | NCBI Taxonomy ID |
| amino_acid_sequence | Full amino acid sequence (FASTA format) |
| gene_sequence | Full gene nucleotide sequence (FASTA format) |

---

### 26. `interactant_polypeptides`
*Interactant ↔ polypeptide junction.*

| Column | Description |
|---|---|
| interactant_id | FK → interactants |
| polypeptide_id | FK → polypeptides |

---

### 27. `polypeptide_attributes`
*Polypeptide synonyms, Pfam protein family domains, and GO classifiers — merged.*

| Column | Description |
|---|---|
| polypeptide_id | FK → polypeptides |
| attr_type | `synonym` / `pfam` / `go_classifier` |
| value | Synonym text; Pfam identifier; GO category |
| value2 | NULL for synonyms; Pfam name; GO description |

---

## Key Statistics

### Step 1 — Full database (`csv_output/`, 707 MB)

| Table | Rows |
|---|---|
| drugs | 19,842 |
| drug_interactions (directed) | 2,911,156 |
| products | 475,225 |
| drug_attributes | ~500,000+ |
| reference_associations | 94,322 |
| references | 43,553 |
| drug_interactants | 34,931 |
| polypeptides | 5,394 unique |
| interactants | 5,449 unique |
| pathways | 48,627 unique |

### Step 2 — Deduplicated DDI pairs (`step2_output/`, 161 MB)

| File | Rows | Note |
|---|---|---|
| drug_interactions_dedup.csv | 1,455,878 | 50% reduction — each unordered pair kept once |

**Columns:** `drugbank_id_a`, `drugbank_id_b`, `description`
(where `drugbank_id_a <= drugbank_id_b` lexicographically; descriptions merged with ` | ` if both directions differ)

### Step 3 — FDA-approved subset (`step3_output/`, 345 MB)

| Metric | Value |
|---|---|
| FDA-approved drugs | 4,795 (of 19,842 total) |
| DDI pairs (both drugs approved, undirected) | 824,249 |
| Products | 473,660 |
| Polypeptides (targets/enzymes/etc.) | 3,439 |
| Pathways containing approved drugs | 48,622 |
| References | 35,721 |

All 27 tables are present in `step3_output/`. Tables filtered by `drugbank_id` (drugs, interactions, products, etc.) contain only approved-drug rows. Lookup tables (categories, pathways, interactants, polypeptides) are filtered to entries actually referenced by the approved-drug rows. The `drug_interactions_dedup.csv` in step3_output/ keeps only pairs where **both** drugs are FDA-approved.

---

## Relationship Diagram (Simplified)

```
drugs ──────────────────── drug_ids
  │                         drug_attributes
  │                         drug_properties
  │                         external_identifiers (entity_type='drug')
  ├── drug_categories ──── categories
  ├── drug_commercial_entities
  ├── dosages / atc_codes / patents
  ├── products / salts ─── external_identifiers (entity_type='salt')
  ├── mixtures / prices
  ├── drug_interactions (self-join → drugs)
  ├── drug_snp_data
  ├── drug_pathways ──── pathways ── pathway_members
  ├── reactions
  ├── drug_interactants ── interactants
  │     (role, actions)        └── interactant_polypeptides ── polypeptides
  │                                       │                      ├── polypeptide_attributes
  │                                       │                      └── external_identifiers
  └── reference_associations ── references
        (interactant_id nullable)
```

---

## Notes for Downstream Use

### For RAG (PharmaBot baseline)
- Primary text fields for embedding: `drugs.description`, `indication`, `mechanism_of_action`, `pharmacodynamics`, `toxicity`
- Interaction text: `drug_interactions.description`
- Combine with `categories`, `atc_codes`, `drug_attributes` (group, synonym) for structured retrieval

### For GNN (link prediction)
- **Edge table (full):** `csv_output/drug_interactions.csv` — 2.9M directed edges, or use `step2_output/drug_interactions_dedup.csv` for 1.45M undirected edges
- **Edge table (approved-only):** `step3_output/drug_interactions_dedup.csv` — 824K undirected edges, cleaner for supervised learning
- **Node features:** `drugs` scalar fields + `drug_properties` (logP, SMILES, MW, etc.) + `drug_attributes` (groups, classifications)
- **Enriched node features:** join `drug_interactants` → `polypeptides` for target/enzyme protein context
- **Negative sampling:** drug pairs absent from `drug_interactions`
- **Recommended starting point:** `step3_output/` — FDA-approved subset reduces noise from withdrawn/experimental compounds
