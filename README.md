# DrugBank XML → CSV Pipeline

**Source:** DrugBank Full Database v5.1 · 19,842 drugs · 2,911,156 DDI pairs
**Python 3.9+ · Dependencies:** `pip install -r requirements.txt` (lxml, pandas)

---

## Pipeline

| Step | Script | Output | Size |
|---|---|---|---|
| 1 — Parse & Validate | `parser/run_all.py` | `data/step1_full/` — 27 CSVs, all drugs | 707 MB · gitignored |
| 2 — Dedup Interactions | `pipeline/step2_dedup_interactions.py` | `data/step2_dedup/` — 1 CSV, undirected DDI | 171 MB · gitignored |
| 3 — FDA-Approved Subset | `pipeline/step3_fda_approved.py` | `data/step3_approved/` — 27 CSVs, 4,795 drugs | 351 MB · **on GitHub** |
| 4 — Build Graph | `pipeline/step4_build_graph.py` | `data/step4_graph/` — node/edge CSVs for GNN | 19 MB · **on GitHub** |

```bash
pip install -r requirements.txt
python parser/run_all.py                       # Step 1: parse + validate (~2 min)
python pipeline/step2_dedup_interactions.py    # Step 2: dedup DDI pairs (~3.5 min)
python pipeline/step3_fda_approved.py          # Step 3: FDA-approved subset (~20 s)
python pipeline/step4_build_graph.py           # Step 4: build graph tensors (~15 s)
```

---

## File Structure

```
├── README.md / .gitignore / requirements.txt
├── parser/                          Step 1 — XML → CSV
│   ├── config.py                    paths, constants, full SCHEMA dict
│   ├── state.py                     ParserState (global dedup counters)
│   ├── utils.py                     XML helpers, CSV writers, ref dedup
│   ├── parse_core.py                → drugs, drug_ids, drug_attributes, drug_properties, external_identifiers
│   ├── parse_references.py          → references, reference_associations
│   ├── parse_commercial.py          → salts, products, drug_commercial_entities, mixtures, prices
│   ├── parse_pharmacological.py     → categories, drug_categories, dosages, atc_codes, patents
│   ├── parse_interactions.py        → drug_interactions, drug_snp_data
│   ├── parse_pathways.py            → pathways, pathway_members, reactions
│   ├── parse_proteins.py            → interactants, drug_interactants, polypeptides,
│   │                                   interactant_polypeptides, polypeptide_attributes
│   ├── main_parser.py               streaming iterparse loop (one drug at a time)
│   ├── validate.py                  16-check validation suite
│   └── run_all.py                   entry point (parse → validate)
├── pipeline/                        Steps 2–4 — post-processing
│   ├── step2_dedup_interactions.py  directed → undirected DDI pairs + interaction_id PK
│   ├── step3_fda_approved.py        filter all tables to FDA-approved drugs
│   └── step4_build_graph.py         build node feature matrix + edge index for GNN
└── data/
    ├── step1_full/                  full parse output             [gitignored]
    ├── step2_dedup/                 undirected DDI pairs          [gitignored]
    ├── step3_approved/              FDA-approved subset           [tracked]
    └── step4_graph/                 GNN-ready node/edge CSVs      [tracked]
```

---

## Tables (27)

> **FK convention:** every `drugbank_id` column is a foreign key to `drugs.drugbank_id` unless noted otherwise.

### 1. `drugs`
*One row per drug. Scalar fields + inlined ClassyFire classification.*

| Column | Description |
|---|---|
| **drugbank_id** | PK — primary DrugBank ID (e.g. DB00001) |
| name | Drug name |
| drug_type | `small molecule` or `biotech` |
| description | Full drug description |
| cas_number | CAS Registry Number |
| unii | FDA Unique Ingredient Identifier |
| average_mass / monoisotopic_mass | Molecular masses (float) |
| state | `solid` / `liquid` / `gas` |
| indication | Therapeutic indications |
| pharmacodynamics | Pharmacodynamics description |
| mechanism_of_action | Mechanism of action |
| toxicity | Toxicity and overdose information |
| metabolism | Metabolic pathway description |
| absorption / half_life / protein_binding | PK parameters |
| route_of_elimination / volume_of_distribution / clearance | PK parameters |
| synthesis_reference | Synthesis reference text |
| fda_label_url / msds_url | External document URLs |
| classification_description | ClassyFire description |
| classification_direct_parent / kingdom / superclass / class / subclass | ClassyFire hierarchy |
| created_date / updated_date | Record timestamps |

### 2. `drug_ids`
*All DrugBank IDs per drug (primary + legacy).*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| legacy_id | ID value (DB#####, BIOD#####, BTD#####, APRD#####, EXPT#####, NUTR#####) |
| is_primary | `True` for the canonical PK used in `drugs` |

### 3. `drug_attributes`
*Catch-all for 9 multi-valued list fields. Filter by `attr_type`.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| attr_type | Type discriminator (see below) |
| value / value2 / value3 | Type-dependent values |

| attr_type | value | value2 | value3 |
|---|---|---|---|
| `group` | `approved` / `withdrawn` / `experimental` / `investigational` / `illicit` / `nutraceutical` / `vet_approved` | — | — |
| `synonym` | synonym text | language code | coder |
| `affected_organism` | organism name | — | — |
| `food_interaction` | description | — | — |
| `sequence` | FASTA string | format | — |
| `ahfs_code` | AHFS code | — | — |
| `pdb_entry` | PDB ID | — | — |
| `classification_alt_parent` | ClassyFire alt parent | — | — |
| `classification_substituent` | ClassyFire substituent | — | — |

### 4. `drug_properties`
*Calculated and experimental physicochemical properties.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| property_class | `calculated` (ChemAxon/ALOGPS) or `experimental` |
| kind | Property name (logP, SMILES, Melting Point, Water Solubility, IUPAC Name, …) |
| value | Property value |
| source | Source tool (ChemAxon, ALOGPS, …) |

### 5. `external_identifiers`
*Cross-database IDs for drugs, salts, polypeptides, and drug web links.*

| Column | Description |
|---|---|
| entity_type | `drug` / `polypeptide` / `salt` / `drug_link` |
| entity_id | PK of the entity (drugbank_id, UniProt ID, or salt_id) |
| resource | Database name (ChEBI, ChEMBL, PubChem, KEGG, BindingDB, PharmGKB, ZINC, RxCUI, HGNC, …) |
| identifier | ID value or URL (for `drug_link`) |

### 6. `references`
*Globally deduplicated bibliography across all drug and protein entries.*

| Column | Description |
|---|---|
| **ref_pk** | PK — auto-increment integer |
| ref_type | `article` / `textbook` / `link` / `attachment` |
| ref_id | DrugBank internal ref ID (e.g. A1, L41539) |
| pubmed_id | PubMed ID (articles) |
| isbn | ISBN (textbooks) |
| title | Title (links/attachments) |
| url | URL (links/attachments) |
| citation | Full citation text |

*Dedup keys: articles → pubmed_id; textbooks → isbn+citation; links → url; attachments → title+url*

### 7. `reference_associations`
*Links each reference to its drug + optional protein context.*

| Column | Description |
|---|---|
| ref_pk | FK → references |
| drugbank_id | FK → drugs |
| interactant_id | FK → interactants — NULL for drug general refs; BE-ID for protein-context refs |

### 8. `salts`
*Salt forms of drugs (DBSALT IDs). Legacy IDs → `external_identifiers` (entity_type=`salt`).*

| Column | Description |
|---|---|
| **salt_id** | PK — DBSALT###### |
| drugbank_id | FK → drugs |
| name / unii / cas_number / inchikey | Identifiers |
| average_mass / monoisotopic_mass | Masses |

### 9. `products`
*Marketed products from FDA NDC, Health Canada DPD, EMA.*

| Column | Description |
|---|---|
| **product_id** | PK — auto-increment |
| drugbank_id | FK → drugs |
| name | Brand/product name |
| labeller | Marketing authorization holder |
| ndc_id / ndc_product_code | FDA NDC identifiers |
| dpd_id | Health Canada DPD ID |
| ema_product_code / ema_ma_number | EMA identifiers |
| started_marketing_on / ended_marketing_on | Date range |
| dosage_form / strength / route | Formulation details |
| fda_application_number | NDA/ANDA number |
| generic / over_the_counter / approved | Boolean flags |
| country | `US` / `Canada` / `EU` |
| source | `FDA NDC` / `DPD` / `EMA` |

### 10. `drug_commercial_entities`
*Packagers, manufacturers, and international brand names — merged with `entity_type`.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| entity_type | `packager` / `manufacturer` / `brand` |
| name | Entity name |
| url | Website URL (packagers/manufacturers) |
| company | Company name (brands) |
| generic_flag | Manufacturer produces generics (manufacturers) |

### 11. `mixtures`

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| name | Mixture/product name |
| ingredients | Ingredient list |
| supplemental_ingredients | Additional ingredients |

### 12. `prices`

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| description | Product description |
| cost / currency / unit | Price details (`USD` or `CAD`) |

### 13. `categories` + 14. `drug_categories`
*Normalized MeSH pharmacological categories.*

| Table | Column | Description |
|---|---|---|
| categories | **category_id** | PK — auto-increment |
| categories | category_name | e.g. "Anticoagulants" |
| categories | mesh_id | MeSH descriptor ID (e.g. D000925) |
| drug_categories | drugbank_id | FK → drugs |
| drug_categories | category_id | FK → categories |

### 15. `dosages`

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| form | Dosage form (Tablet, Injection, Powder, …) |
| route | Route (Oral, Intravenous, …) |
| strength | e.g. "50 mg" |

### 16. `atc_codes`
*WHO ATC classification with full 4-level hierarchy.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| atc_code | Full code (e.g. B01AE02) |
| l1_code / l1_name | Most specific subgroup |
| l2_code / l2_name | Pharmacological subgroup |
| l3_code / l3_name | Pharmacological/therapeutic group |
| l4_code / l4_name | Anatomical main group (e.g. B / "BLOOD AND BLOOD FORMING ORGANS") |

### 17. `patents`

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| number / country | Patent identifier |
| approved_date / expires_date | Date range |
| pediatric_extension | Boolean — pediatric exclusivity |

### 18. `drug_interactions`
*DDI records — the core edge table. **⚠ No `interaction_id` PK in step1_full yet** (planned for next parser run). Use the deduplicated version in step2/step3 which includes `interaction_id`.*

| Column | Description |
|---|---|
| drugbank_id | Source drug FK |
| interacting_drugbank_id | Target drug FK |
| description | Interaction description text |

> Both directions stored (A→B and B→A). Use `drug_interactions_dedup.csv` from step2/step3 for undirected edges with `interaction_id` PK.

### 19. `drug_snp_data`
*SNP pharmacogenomics — effects and adverse reactions merged with `snp_type`.*

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| snp_type | `effect` or `adverse_reaction` |
| protein_name / gene_symbol / uniprot_id | Affected protein |
| rs_id / allele / defining_change | Variant details |
| adverse_reaction | Reaction name (adverse_reactions only) |
| description / pubmed_id | Clinical evidence |

### 20. `pathways` + 21. `pathway_members`
*SMPDB pathway entities and their drug/enzyme members.*

| Table | Column | Description |
|---|---|---|
| pathways | **smpdb_id** | PK — SMPDB pathway ID |
| pathways | name | Pathway name |
| pathways | category | e.g. Drug metabolism, Signaling |
| pathway_members | smpdb_id | FK → pathways |
| pathway_members | member_type | `drug` or `enzyme` |
| pathway_members | member_id | DrugBank ID (drug) or UniProt ID (enzyme) |
| pathway_members | member_name | Name (drugs only) |

### 22. `reactions`
*Metabolic reaction records. Enzymes are pipe-serialized in the `enzymes` field.*

| Column | Description |
|---|---|
| **reaction_id** | PK — auto-increment |
| drugbank_id | FK → drugs |
| sequence | Reaction step ID |
| left_element_id / left_element_name | Substrate |
| right_element_id / right_element_name | Product |
| enzymes | Pipe-delimited triples: `db_id\|name\|uniprot_id\|\|…` |

### 23. `interactants` + 24. `drug_interactants`
*Binding entities (targets, enzymes, carriers, transporters) and their drug links.*

| Table | Column | Description |
|---|---|---|
| interactants | **interactant_id** | PK — BE-ID (e.g. BE0000048) |
| interactants | name | Protein name |
| interactants | organism | Source organism |
| drug_interactants | drugbank_id | FK → drugs |
| drug_interactants | interactant_id | FK → interactants |
| drug_interactants | role | `target` / `enzyme` / `carrier` / `transporter` |
| drug_interactants | position | Order within role group |
| drug_interactants | known_action | `yes` / `no` / `unknown` |
| drug_interactants | actions | Pipe-delimited actions (e.g. `inhibitor\|substrate`) |
| drug_interactants | inhibition_strength / induction_strength | Enzyme-only fields |

### 25. `polypeptides`
*UniProt protein records, globally deduplicated.*

| Column | Description |
|---|---|
| **polypeptide_id** | PK — UniProt ID (e.g. P00734) |
| source | `Swiss-Prot` or `TrEMBL` |
| name | Protein name |
| general_function / specific_function | Functional descriptions |
| gene_name | HGNC gene symbol |
| locus / cellular_location / chromosome_location | Localization |
| transmembrane_regions / signal_regions | Structural features |
| theoretical_pi / molecular_weight | Biophysical properties |
| organism / ncbi_taxonomy_id | Source organism |
| amino_acid_sequence / gene_sequence | Full FASTA sequences |

### 26. `interactant_polypeptides`
*Junction between binding entities and UniProt polypeptides.*

| Column | Description |
|---|---|
| interactant_id | FK → interactants |
| polypeptide_id | FK → polypeptides |

### 27. `polypeptide_attributes`
*Synonyms, Pfam domains, and GO classifiers — merged with `attr_type`.*

| Column | Description |
|---|---|
| polypeptide_id | FK → polypeptides |
| attr_type | `synonym` / `pfam` / `go_classifier` |
| value | Synonym text; Pfam ID; GO category |
| value2 | — (synonyms); Pfam name; GO description |

---

## `drug_interactions_dedup.csv` (step2 & step3)
*Undirected DDI pairs with integer PK. Present in `data/step2_dedup/` and `data/step3_approved/`.*

| Column | Description |
|---|---|
| **interaction_id** | PK — auto-increment integer (1-based) |
| drugbank_id_a | Drug A (lexicographically smaller ID) |
| drugbank_id_b | Drug B (lexicographically larger ID) |
| description | Merged description (both directions joined with ` \| ` if they differ) |

---

## Key Statistics

| | Full (step1) | Approved-only (step3) |
|---|---|---|
| Drugs | 19,842 | **4,795** |
| DDI pairs (directed, step1) | 2,911,156 | — |
| DDI pairs (undirected, step2/3) | 1,455,878 | **824,249** |
| Products | 475,225 | 473,660 |
| Polypeptides | 5,394 | 3,439 |
| Interactants (BE-IDs) | 5,449 | 3,458 |
| Pathways | 48,627 | 48,622 |
| References | 43,553 | 35,721 |
| Drug–protein links | 34,931 | 20,700 |

> **FDA-approved** = drug has at least one `group=approved` entry in `drug_attributes`. DrugBank's "approved" label covers FDA, EMA, Health Canada, and other major agencies — not exclusively FDA.

---

## Relationship Map

```
drugs ─────────────────────── drug_ids
  │                            drug_attributes   (group, synonym, food_interaction, …)
  │                            drug_properties   (logP, SMILES, MW, …)
  │                            external_identifiers (entity_type='drug'/'drug_link')
  ├── drug_categories ──────── categories
  ├── products / salts ──────── external_identifiers (entity_type='salt')
  ├── drug_commercial_entities (packagers, manufacturers, brands)
  ├── dosages / atc_codes / patents
  ├── mixtures / prices
  ├── drug_interactions ──────── (self-join → drugs)  →  drug_interactions_dedup (step2/3)
  ├── drug_snp_data
  ├── reactions
  ├── pathway_members ──────── pathways
  ├── drug_interactants ──────── interactants
  │     (role, actions)              └── interactant_polypeptides ── polypeptides
  │                                                                    ├── polypeptide_attributes
  │                                                                    └── external_identifiers
  └── reference_associations ── references
        (interactant_id nullable)
```

---

## Step 4 — Graph Build (`data/step4_graph/`)

Built from `data/step3_approved/`. All 4,795 FDA-approved drugs become nodes; all 824,249 undirected DDI pairs become edges.

| File | Rows | Description |
|---|---|---|
| `node_mapping.csv` | 4,795 | `node_idx`, `drugbank_id`, `name` — integer index ↔ drug |
| `node_features.csv` | 4,795 | `node_idx` + 70 standardized features (mean=0, std=1) |
| `node_features_raw.csv` | 4,795 | Same 70 features, unscaled original values |
| `edge_index.csv` | 824,249 | `src_idx`, `dst_idx`, `interaction_id` |
| `feature_names.json` | — | Feature name list + group labels |

**Feature groups (70 total):**

| Group | Features | Source |
|---|---|---|
| A — Masses (2) | `average_mass`, `monoisotopic_mass` | `drugs.csv` |
| B — Type & state (4) | `is_biotech`, `state_solid/liquid/gas` | `drugs.csv` |
| C — Calculated props (17) | logP, logS, MW, HBD, HBA, RotB, PSA, charge, rings, bioavailability, Rule of Five, Ghose filter, MDDR-like rule, refractivity, polarizability, pKa acid/basic | `drug_properties.csv` (`property_class='calculated'`, ChemAxon/ALOGPS) |
| D — Experimental props (6) | logP, logS, melting point, boiling point, water solubility, pKa | `drug_properties.csv` (`property_class='experimental'`, lab-measured) |
| E — Group flags (5) | `is_withdrawn`, `is_investigational`, `is_vet_approved`, `is_nutraceutical`, `is_illicit` | `drug_attributes.csv` (`attr_type='group'`) |
| F — Counts (10) | n_targets, n_enzymes, n_carriers, n_transporters, n_categories, n_atc_codes, n_patents, n_products, n_food_interactions, n_synonyms | Counted from `drug_interactants`, `drug_categories`, `atc_codes`, `patents`, `products`, `drug_attributes` |
| G — ATC anatomical (14) | `atc_A` … `atc_V` one-hot (A=Alimentary, B=Blood, C=Cardiovascular, J=Anti-infectives, N=Nervous system, …) | `atc_codes.csv` (`l4_code`) |
| H — Classification (12) | `kingdom_organic`, `kingdom_inorganic`; top-10 superclass one-hot | `drugs.csv` (`classification_kingdom`, `classification_superclass` — ClassyFire) |

Missing continuous values are imputed with the column median. Binary/count/one-hot missing values are filled with 0.

---

## Downstream Use

**GNN (link prediction)**
- **Start here:** `data/step4_graph/` — ready-to-load node features + edge index
- **Edges:** `edge_index.csv` — 824K undirected pairs as integer (src, dst) indices
- **Node features:** `node_features.csv` — 4,795 × 70, already standardized
- **Negative sampling:** drug node pairs absent from `edge_index.csv`
- **Protein context (optional enrichment):** join `step3_approved/drug_interactants.csv` → `polypeptides.csv` for target/enzyme embeddings

**RAG (PharmaBot)**
- **Text fields:** `drugs.description`, `indication`, `mechanism_of_action`, `pharmacodynamics`, `toxicity`
- **Interaction text:** `drug_interactions_dedup.csv` → `description`
- **Structured filters:** `categories`, `atc_codes`, `drug_attributes` (group, synonym)
