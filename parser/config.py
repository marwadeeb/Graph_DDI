"""
config.py — shared paths and constants for DrugBank XML parsing pipeline.
"""
import os

PYTHON_EXE = r"C:\Users\LENOVO\AppData\Local\Programs\Python\Python39\python.exe"

# Input
XML_PATH = r"D:\DDI\drugbank_all_full_database.xml\full database.xml"
XSD_PATH = r"C:\Users\LENOVO\Downloads\drugbank (1).xsd"
WORKING_DIR = r"D:\DDI\drugbank_all_full_database.xml"

# Output
OUTPUT_DIR = os.path.join(WORKING_DIR, "data", "step1_full")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# XML namespace
NS = "http://www.drugbank.ca"
NP = f"{{{NS}}}"   # "{http://www.drugbank.ca}" — prefix used in find/findall calls

# Progress reporting interval (drugs)
PROGRESS_EVERY = 1000

# All 27 table names
TABLES = [
    "drugs",
    "drug_ids",
    "drug_attributes",
    "drug_properties",
    "external_identifiers",
    "references",
    "reference_associations",
    "salts",
    "products",
    "drug_commercial_entities",
    "mixtures",
    "prices",
    "categories",
    "drug_categories",
    "dosages",
    "atc_codes",
    "patents",
    "drug_interactions",
    "drug_snp_data",
    "pathways",
    "pathway_members",
    "reactions",
    "interactants",
    "drug_interactants",
    "polypeptides",
    "interactant_polypeptides",
    "polypeptide_attributes",
]

# Column definitions for every table
SCHEMA = {
    "drugs": [
        "drugbank_id", "name", "drug_type", "description", "cas_number", "unii",
        "average_mass", "monoisotopic_mass", "state",
        "indication", "pharmacodynamics", "mechanism_of_action", "toxicity",
        "metabolism", "absorption", "half_life", "protein_binding",
        "route_of_elimination", "volume_of_distribution", "clearance",
        "synthesis_reference", "fda_label_url", "msds_url",
        "classification_description", "classification_direct_parent",
        "classification_kingdom", "classification_superclass",
        "classification_class", "classification_subclass",
        "created_date", "updated_date",
    ],
    "drug_ids": ["drugbank_id", "legacy_id", "is_primary"],
    "drug_attributes": ["drugbank_id", "attr_type", "value", "value2", "value3"],
    "drug_properties": ["drugbank_id", "property_class", "kind", "value", "source"],
    "external_identifiers": ["entity_type", "entity_id", "resource", "identifier"],
    "references": ["ref_pk", "ref_type", "ref_id", "pubmed_id", "isbn", "title", "url", "citation"],
    "reference_associations": ["ref_pk", "drugbank_id", "interactant_id"],
    "salts": ["salt_id", "drugbank_id", "name", "unii", "cas_number", "inchikey",
              "average_mass", "monoisotopic_mass"],
    "products": [
        "product_id", "drugbank_id", "name", "labeller", "ndc_id", "ndc_product_code",
        "dpd_id", "ema_product_code", "ema_ma_number",
        "started_marketing_on", "ended_marketing_on",
        "dosage_form", "strength", "route", "fda_application_number",
        "generic", "over_the_counter", "approved", "country", "source",
    ],
    "drug_commercial_entities": [
        "drugbank_id", "entity_type", "name", "url", "company", "generic_flag",
    ],
    "mixtures": ["drugbank_id", "name", "ingredients", "supplemental_ingredients"],
    "prices": ["drugbank_id", "description", "cost", "currency", "unit"],
    "categories": ["category_id", "category_name", "mesh_id"],
    "drug_categories": ["drugbank_id", "category_id"],
    "dosages": ["drugbank_id", "form", "route", "strength"],
    "atc_codes": [
        "drugbank_id", "atc_code",
        "l1_code", "l1_name", "l2_code", "l2_name",
        "l3_code", "l3_name", "l4_code", "l4_name",
    ],
    "patents": ["drugbank_id", "number", "country", "approved_date", "expires_date",
                "pediatric_extension"],
    "drug_interactions": ["drugbank_id", "interacting_drugbank_id", "description"],
    "drug_snp_data": [
        "drugbank_id", "snp_type", "protein_name", "gene_symbol", "uniprot_id",
        "rs_id", "allele", "defining_change", "adverse_reaction", "description", "pubmed_id",
    ],
    "pathways": ["smpdb_id", "name", "category"],
    "pathway_members": ["smpdb_id", "member_type", "member_id", "member_name"],
    "reactions": [
        "reaction_id", "drugbank_id", "sequence",
        "left_element_id", "left_element_name",
        "right_element_id", "right_element_name",
        "enzymes",
    ],
    "interactants": ["interactant_id", "name", "organism"],
    "drug_interactants": [
        "drugbank_id", "interactant_id", "role", "position", "known_action",
        "actions", "inhibition_strength", "induction_strength",
    ],
    "polypeptides": [
        "polypeptide_id", "source", "name", "general_function", "specific_function",
        "gene_name", "locus", "cellular_location", "transmembrane_regions",
        "signal_regions", "theoretical_pi", "molecular_weight",
        "chromosome_location", "organism", "ncbi_taxonomy_id",
        "amino_acid_sequence", "gene_sequence",
    ],
    "interactant_polypeptides": ["interactant_id", "polypeptide_id"],
    "polypeptide_attributes": ["polypeptide_id", "attr_type", "value", "value2"],
}
