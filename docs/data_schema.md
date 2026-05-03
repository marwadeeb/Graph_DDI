# Data Schema — DDI Checker

27 normalised tables produced by `parser/run_all.py` from the DrugBank XML.

---

## Key Statistics

| | Full (step1) | Approved-only (step3) |
|---|---|---|
| Drugs | 19,842 | **4,795** |
| DDI pairs (directed) | 2,911,156 | — |
| DDI pairs (undirected) | 1,455,878 | **824,249** |
| Products | 475,225 | 473,660 |
| Polypeptides | 5,394 | 3,439 |
| Interactants (BE-IDs) | 5,449 | 3,458 |
| Pathways | 48,627 | 48,622 |
| References | 43,553 | 35,721 |
| Drug-protein links | 34,931 | 20,700 |

---

## FK Convention

Every `drugbank_id` column is a FK to `drugs.drugbank_id` unless noted.

---

## Tables

### 1. `drugs`
One row per drug. Scalar fields + inlined ClassyFire classification.

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
| classification_description | ClassyFire description |
| classification_direct_parent / kingdom / superclass / class / subclass | ClassyFire hierarchy |
| created_date / updated_date | Record timestamps |

### 2. `drug_ids`

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| legacy_id | ID value (DB#####, BIOD#####, BTD#####, APRD#####, EXPT#####, NUTR#####) |
| is_primary | `True` for the canonical PK used in `drugs` |

### 3. `drug_attributes`
Catch-all for 9 multi-valued list fields. Filter by `attr_type`.

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

| Column | Description |
|---|---|
| drugbank_id | FK → drugs |
| property_class | `calculated` (ChemAxon/ALOGPS) or `experimental` |
| kind | Property name (logP, SMILES, Melting Point, Water Solubility, IUPAC Name, …) |
| value | Property value |
| source | Source tool (ChemAxon, ALOGPS, …) |

### 5. `external_identifiers`

| Column | Description |
|---|---|
| entity_type | `drug` / `polypeptide` / `salt` / `drug_link` |
| entity_id | PK of the entity |
| resource | Database name (ChEBI, ChEMBL, PubChem, KEGG, BindingDB, PharmGKB, ZINC, RxCUI, HGNC, …) |
| identifier | ID value or URL |

### 6. `references` + 7. `reference_associations`
Globally deduplicated bibliography. Dedup keys: articles → pubmed_id; textbooks → isbn+citation; links → url; attachments → title+url.

### 8. `salts`, 9. `products`, 10. `drug_commercial_entities`, 11. `mixtures`, 12. `prices`
Standard commercial/formulation tables.

### 13. `categories` + 14. `drug_categories`
Normalized MeSH pharmacological categories.

### 15. `dosages`, 16. `atc_codes`, 17. `patents`
`atc_codes` has full 4-level hierarchy: `l1_code/l1_name` … `l4_code/l4_name`.

### 18. `drug_interactions`
Core edge table (directed). Both A→B and B→A stored. Use `drug_interactions_dedup.csv` for undirected edges.

| Column | Description |
|---|---|
| drugbank_id | Source drug FK |
| interacting_drugbank_id | Target drug FK |
| description | Interaction description text |

### 19. `drug_snp_data`, 20. `pathways`, 21. `pathway_members`, 22. `reactions`
Pharmacogenomics (SNP), SMPDB pathways, metabolic reactions.

### 23. `interactants` + 24. `drug_interactants`
Binding entities (targets, enzymes, carriers, transporters).

| Column | Description |
|---|---|
| interactant_id | BE-ID (e.g. BE0000048) |
| role | `target` / `enzyme` / `carrier` / `transporter` |
| known_action | `yes` / `no` / `unknown` |
| actions | Pipe-delimited (e.g. `inhibitor\|substrate`) |
| inhibition_strength / induction_strength | Enzyme-only |

### 25. `polypeptides`, 26. `interactant_polypeptides`, 27. `polypeptide_attributes`
UniProt protein records, globally deduplicated. Attributes: synonyms, Pfam domains, GO classifiers.

---

## `drug_interactions_dedup.csv`
Undirected DDI pairs with integer PK. Present in `step2_dedup/` and `step3_approved/`.

| Column | Description |
|---|---|
| **interaction_id** | PK — auto-increment integer (1-based) |
| drugbank_id_a | Drug A (lexicographically smaller ID) |
| drugbank_id_b | Drug B (lexicographically larger ID) |
| description | Merged description (both directions joined with ` \| ` if they differ) |
