# Import required libraries
import requests
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set, Union, FrozenSet
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import networkx as nx
import re
import time
from urllib.parse import quote
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oncology_miner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize NLP model
logger.info("Initializing NLP model...")
try:
    nlp = spacy.load("en_core_sci_sm")
    logger.info("Loaded en_core_sci_sm model successfully.")
except OSError:
    logger.warning("Scientific model not found, falling back to medium model...")
    try:
        nlp = spacy.load("en_core_web_md")
        logger.info("Loaded en_core_web_md model successfully.")
    except OSError:
        logger.warning("Medium model not found, falling back to small English model...")
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded en_core_web_sm model successfully.")
        logger.info("Note: For better results, install scispaCy models:")
        logger.info("pip install scispacy")
        logger.info("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz")

# Constants
CACHE_DIR = "./data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
logger.info(f"Using cache directory: {CACHE_DIR}")

# Configure requests session
session = requests.Session()
session.headers.update({'User-Agent': 'OncologyLiteratureMiner/1.0'})

class LiteratureCollector:
    """Handles collection of literature from various sources"""

    def __init__(self):
        self.base_urls = {
            'pubmed': "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            'clinical_trials': "https://clinicaltrials.gov/api/v2/studies",
            'google_patents': "https://patents.googleapis.com/v1/patents:search"
        }
        self.max_retries = 3
        self.retry_delay = 2
        logger.info("LiteratureCollector initialized.")

    def _make_request(self, url: str, params: Dict, is_xml: bool = False):
        for attempt in range(self.max_retries):
            try:
                resp = session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                if is_xml and not resp.content.strip():
                    raise ValueError("Empty XML response")
                return resp
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Request failed: {e}")
                    return None
                time.sleep(self.retry_delay)
    
    def build_pubmed_query(self, drugs: List[str] = None, 
                         biomarkers: List[str] = None,
                         cancers: List[str] = None,
                         genes: List[str] = None,
                         date_range: Tuple[str, str] = None) -> str:
        """
        Build a sophisticated PubMed query from components
        Args:
            drugs: List of drug names
            biomarkers: List of biomarkers
            cancers: List of cancer types
            genes: List of genes
            date_range: Tuple of (start_date, end_date) in YYYY/MM/DD format
        Returns:
            PubMed-compatible query string
        """
        query_parts = []
        
        # Add terms with field tags
        if drugs:
            drug_terms = [f'"{drug}"[Title/Abstract]' for drug in drugs]
            query_parts.append(f"({' OR '.join(drug_terms)})")
            
        if biomarkers:
            bio_terms = [f'"{bm}"[Title/Abstract]' for bm in biomarkers]
            query_parts.append(f"({' OR '.join(bio_terms)})")
            
        if cancers:
            cancer_terms = [f'"{cancer}"[Title/Abstract]' for cancer in cancers]
            query_parts.append(f"({' OR '.join(cancer_terms)})")
            
        if genes:
            gene_terms = [f'"{gene}"[Title/Abstract]' for gene in genes]
            query_parts.append(f"({' OR '.join(gene_terms)})")
            
        # Add date filter if provided
        if date_range:
            start, end = date_range
            query_parts.append(f'("{start}"[Date - Publication] : "{end}"[Date - Publication])')
            
        # Combine with AND
        return ' AND '.join(query_parts) if query_parts else 'cancer'

    def search_pubmed(self, query: str, max_results: int = 1000) -> List[Dict]:
        """
        Search PubMed and return a list of article metadata.
        """
        # 1. Define search parameters up front
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(max_results, 10000),
            'retmode': 'json',
        }
        # Only add the api_key if the env var is set
        api_key = os.getenv('NCBI_API_KEY')
        if api_key:
            search_params['api_key'] = api_key

        # 2. Execute the search
        logger.info(f"Searching PubMed for: '{query}'")
        response = self._make_request(self.base_urls['pubmed'] + 'esearch.fcgi', search_params)
        if not response:
            logger.error("PubMed search request failed")
            return []

        # 3. Parse PMIDs
        data = response.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        if not pmids:
            logger.info("No articles found matching query")
            return []

        logger.info(f"Found {len(pmids)} articles. Fetching details...")

        # 4. Fetch details in batches
        articles = []
        for i in range(0, len(pmids), 100):
            batch = pmids[i:i+100]
            fetch_params = {
                'db': 'pubmed',
                'id': ",".join(batch),
                'retmode': 'xml',
                'api_key': os.getenv('NCBI_API_KEY')
            }
            fetch_resp = self._make_request(self.base_urls['pubmed'] + 'efetch.fcgi',
                                          fetch_params,
                                          is_xml=True)
            if not fetch_resp:
                continue

            # 5. Parse XML
            try:
                root = ET.fromstring(fetch_resp.content)
            except ET.ParseError:
                content = fetch_resp.content
                if content.startswith(b'<?xml'):
                    content = content.split(b'?>', 1)[1]
                root = ET.fromstring(content)

            for art in root.findall('.//PubmedArticle'):
                try:
                    pmid = art.findtext('.//PMID')
                    title = art.findtext('.//ArticleTitle')
                    abstract = ' '.join([t.text for t in art.findall('.//AbstractText') if t.text])
                    journal = art.findtext('.//Journal/Title')
                    year = art.findtext('.//PubDate/Year')
                    month = art.findtext('.//PubDate/Month')
                    day = art.findtext('.//PubDate/Day')
                    pub_date = "-".join(filter(None, [year, month, day]))
                    doi = art.findtext('.//ArticleId[@IdType="doi"]')
                    pmcid = art.findtext('.//ArticleId[@IdType="pmc"]')
                    authors = [f"{a.findtext('LastName')} {a.findtext('ForeName')}"
                            for a in art.findall('.//Author')
                            if a.findtext('LastName') and a.findtext('ForeName')]
                    articles.append({
                        'pmid': pmid,
                        'title': title,
                        'abstract': abstract,
                        'journal': journal,
                        'pub_date': pub_date,
                        'doi': doi,
                        'pmcid': pmcid,
                        'authors': authors
                    })
                except Exception as e:
                    logger.warning(f"Error parsing article XML: {e}")
                    continue

            time.sleep(0.5)

        # 6. Cache and return
        cache_file = os.path.join(CACHE_DIR, f"pubmed_{hash(query)}.json")
        if articles:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(articles, f, indent=2)
                logger.info(f"Cached {len(articles)} articles to {cache_file}")
            except Exception as e:
                logger.error(f"Could not write cache: {e}")

        return articles

    def search_clinical_trials(self, condition: str, intervention: str = None, max_results: int = 100) -> List[Dict]:
        """
        Search ClinicalTrials.gov for oncology trials
        Args:
            condition: Cancer condition to search for
            intervention: Optional intervention filter
            max_results: Maximum number of results to return
        Returns:
            List of trial metadata dictionaries with keys:
            - nct_id: ClinicalTrials.gov identifier
            - title: Brief title
            - official_title: Official title
            - description: Brief summary
            - conditions: List of conditions
            - interventions: List of interventions
            - phase: Trial phase
            - status: Overall status
            - start_date: Start date
            - completion_date: Completion date
            - study_type: Type of study
            - enrollment: Enrollment count
        """
        cache_key = f"clinical_{condition}_{intervention if intervention else ''}"
        cache_file = os.path.join(CACHE_DIR, f"{hash(cache_key)}.json")

        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data:
                        logger.info(f"Loaded {len(cached_data)} trials from cache")
                        return cached_data
            except Exception as e:
                logger.error(f"Error loading cache: {e}")

        # Build query parameters
        params = {
            'query.cond': condition.replace(' ', '+'),
            'format': 'json',
            'pageSize': min(max_results, 1000)
        }
        if intervention:
            params['query.intr'] = intervention.replace(' ', '+')

        # Updated ClinicalTrials.gov API endpoint (v2)
        base_url = "https://clinicaltrials.gov/api/v2/studies"

        logger.info(f"Querying ClinicalTrials.gov for: {condition}")
        if intervention:
            logger.info(f"With intervention: {intervention}")

        try:
            response = session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            processed_trials = []
            for study in data.get('studies', []):
                try:
                    protocol = study.get('protocolSection', {})
                    identification = protocol.get('identificationModule', {})
                    status = protocol.get('statusModule', {})
                    design = protocol.get('designModule', {})
                    description = protocol.get('descriptionModule', {})
                    arms = protocol.get('armsInterventionsModule', {})
                    
                    processed_trial = {
                        'nct_id': identification.get('nctId'),
                        'title': identification.get('briefTitle'),
                        'official_title': identification.get('officialTitle'),
                        'description': description.get('briefSummary'),
                        'conditions': [c.get('name') for c in design.get('conditions', [])],
                        'interventions': [i.get('name') for i in arms.get('interventions', [])],
                        'phase': design.get('phase'),
                        'status': status.get('overallStatus'),
                        'start_date': status.get('startDate'),
                        'completion_date': status.get('completionDate'),
                        'study_type': design.get('studyType'),
                        'enrollment': design.get('enrollmentInfo', {}).get('count')
                    }
                    processed_trials.append(processed_trial)
                except Exception as e:
                    logger.warning(f"Error processing trial: {str(e)}")
                    continue

            # Cache results
            if processed_trials:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(processed_trials, f, indent=2)
                    logger.info(f"Cached {len(processed_trials)} trials")
                except Exception as e:
                    logger.error(f"Could not cache results: {str(e)}")

            return processed_trials

        except Exception as e:
            logger.error(f"Error accessing ClinicalTrials.gov API: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"API response: {e.response.text[:500]}")
            return []

    def search_google_patents(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search Google Patents for oncology-related patents
        Args:
            query: Search query
            max_results: Maximum number of results
        Returns:
            List of patent dictionaries
        """
        cache_key = f"patents_{query}"
        cache_file = os.path.join(CACHE_DIR, f"{hash(cache_key)}.json")
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading patent cache: {e}")
                
        params = {
            'query': query,
            'pageSize': min(max_results, 100),
            'key': os.getenv('GOOGLE_PATENTS_API_KEY', '')  # Add your API key
        }
        
        try:
            logger.info(f"Searching Google Patents for: {query}")
            response = session.get(self.base_urls['google_patents'], params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            patents = []
            for patent in data.get('patents', []):
                try:
                    patents.append({
                        'patent_id': patent.get('patentId'),
                        'title': patent.get('title'),
                        'abstract': patent.get('abstractText'),
                        'filing_date': patent.get('filingDate'),
                        'publication_date': patent.get('publicationDate'),
                        'inventors': [inv.get('name') for inv in patent.get('inventors', [])],
                        'assignees': [ass.get('name') for ass in patent.get('assignees', [])],
                        'claims': [claim.get('text') for claim in patent.get('claims', [])]
                    })
                except Exception as e:
                    logger.warning(f"Error processing patent: {e}")
                    continue
                    
            # Cache results
            if patents:
                with open(cache_file, 'w') as f:
                    json.dump(patents, f, indent=2)
                    
            return patents
            
        except Exception as e:
            logger.error(f"Error accessing Google Patents API: {e}")
            return []

class TextProcessor:
    """Handles text preprocessing and entity recognition"""

    def __init__(self):
        logger.info("Initializing TextProcessor with oncology dictionaries...")
        self.cancer_types = self._load_cancer_types()
        self.drug_lexicon = self._load_drug_lexicon()
        self.biomarkers = self._load_biomarkers()
        self.genes = self._load_genes()
        self.mutations = self._load_mutations()
        
        logger.info(f"Loaded {len(self.cancer_types)} cancer types, {len(self.drug_lexicon)} drugs, "
                   f"{len(self.biomarkers)} biomarkers, {len(self.genes)} genes, "
                   f"{len(self.mutations)} mutations")
        
    def _load_cancer_types(self) -> Set[str]:
        """Load cancer types from MeSH/ICD-10"""
        return {
            'breast cancer', 'lung cancer', 'colorectal cancer', 'prostate cancer',
            'melanoma', 'leukemia', 'lymphoma', 'glioblastoma', 'pancreatic cancer',
            'ovarian cancer', 'bladder cancer', 'hepatocellular carcinoma',
            'non-small cell lung cancer', 'triple-negative breast cancer',
            'renal cell carcinoma', 'gastric cancer', 'esophageal cancer',
            'head and neck cancer', 'cervical cancer', 'endometrial cancer'
        }
        
    def _load_drug_lexicon(self) -> Dict[str, str]:
        """Load drug names and categories"""
        return {
            'trastuzumab': 'targeted therapy', 'pertuzumab': 'targeted therapy',
            'pembrolizumab': 'immunotherapy', 'nivolumab': 'immunotherapy',
            'ipilimumab': 'immunotherapy', 'atezolizumab': 'immunotherapy',
            'cisplatin': 'chemotherapy', 'carboplatin': 'chemotherapy',
            'paclitaxel': 'chemotherapy', 'docetaxel': 'chemotherapy',
            'doxorubicin': 'chemotherapy', 'cyclophosphamide': 'chemotherapy',
            'tamoxifen': 'hormone therapy', 'letrozole': 'hormone therapy',
            'olaparib': 'targeted therapy', 'rucaparib': 'targeted therapy',
            'erlotinib': 'targeted therapy', 'gefitinib': 'targeted therapy',
            'osimertinib': 'targeted therapy', 'cetuximab': 'targeted therapy'
        }
        
    def _load_biomarkers(self) -> Set[str]:
        """Load known biomarkers"""
        return {
            'PD-L1', 'HER2', 'ER', 'PR', 'KRAS', 'NRAS', 'BRCA1', 'BRCA2',
            'EGFR', 'ALK', 'BRAF', 'MSI', 'TMB', 'TP53', 'PTEN', 'PIK3CA',
            'ROS1', 'MET', 'NTRK', 'FGFR', 'RET', 'AR', 'CDK4', 'CDK6'
        }
        
    def _load_genes(self) -> Set[str]:
        """Load cancer-related genes"""
        return {
            'EGFR', 'HER2', 'KRAS', 'NRAS', 'BRAF', 'ALK', 'ROS1', 'MET',
            'BRCA1', 'BRCA2', 'TP53', 'PTEN', 'PIK3CA', 'MYC', 'CDKN2A',
            'RB1', 'APC', 'NOTCH1', 'JAK2', 'STAT3', 'NF1', 'NF2', 'VHL'
        }
        
    def _load_mutations(self) -> Set[str]:
        """Load common cancer mutations"""
        return {
            'EGFR L858R', 'EGFR T790M', 'KRAS G12D', 'KRAS G12V',
            'BRAF V600E', 'PIK3CA H1047R', 'TP53 R175H', 'BRCA1 5382insC',
            'ALK EML4-ALK', 'HER2 amplification', 'MET exon 14 skipping'
        }
        
    def clean_text(self, text: str) -> str:
        """
        Advanced text cleaning pipeline
        Args:
            text: Input text
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove references/citations
        text = re.sub(r'\[[\d,]+\]', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)
        
        return text
        
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text:
            return ""
            
        # Remove special characters, extra whitespace, and normalize
        text = re.sub(r'[^\w\s-]', ' ', text.lower())  # Keep words, spaces, and hyphens
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def normalize_entity(self, entity: str, entity_type: str) -> str:
        """
        Normalize entity names to standard forms
        Args:
            entity: Entity text
            entity_type: Type of entity
        Returns:
            Normalized entity name
        """
        if not entity:
            return ""
            
        # Common normalizations
        entity = entity.strip().lower()
        
        # Type-specific normalizations
        if entity_type == 'drug':
            # Remove trade names in parentheses
            entity = re.sub(r'\(.*?\)', '', entity)
            # Standardize endings
            if entity.endswith('mab'):
                entity = entity[:-3] + 'mab'  # Standardize monoclonal antibodies
            elif entity.endswith('nib'):
                entity = entity[:-3] + 'nib'  # Standardize kinase inhibitors
                
        elif entity_type in ['gene', 'biomarker']:
            # Standardize gene names
            entity = entity.upper()
            # Remove common prefixes
            entity = re.sub(r'^(gene|protein)\s+', '', entity)
            
        elif entity_type == 'cancer_type':
            # Standardize cancer names
            if not entity.endswith(' cancer'):
                entity = f"{entity} cancer"
                
        return entity.strip()
        
    def link_entities_to_databases(self, entities: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """
        Link entities to standard databases
        Args:
            entities: Dictionary of entity lists
        Returns:
            Dictionary with linked entities including database IDs
        """
        linked_entities = {k: [] for k in entities.keys()}
        
        # Mock database links - in practice you'd use APIs or local databases
        db_mappings = {
            'drugs': {
                'trastuzumab': {'drugbank': 'DB00072', 'chebi': 'CHEBI:72447'},
                'pembrolizumab': {'drugbank': 'DB09037', 'chebi': 'CHEBI:90551'}
            },
            'genes': {
                'BRCA1': {'entrez': '672', 'hgnc': '1100'},
                'EGFR': {'entrez': '1956', 'hgnc': '3236'}
            },
            'cancer_types': {
                'breast cancer': {'mesh': 'D001943', 'icd10': 'C50'},
                'lung cancer': {'mesh': 'D002283', 'icd10': 'C34'}
            }
        }
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                norm_entity = self.normalize_entity(entity, entity_type)
                db_links = db_mappings.get(entity_type, {}).get(norm_entity, {})
                
                linked_entities[entity_type].append({
                    'text': entity,
                    'normalized': norm_entity,
                    'db_links': db_links
                })
                
        return linked_entities
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract biomedical entities from text
        Args:
            text: Input text to process
        Returns:
            Dictionary of entity types and their values
        """
        if not text:
            return {}
            
        text = self.preprocess_text(text)
        doc = nlp(text)
        
        entities = {
            'cancer_types': [],
            'drugs': [],
            'biomarkers': [],
            'genes': [],
            'mutations': [],
            'other_entities': []
        }
        
        # Rule-based extraction
        for cancer in self.cancer_types:
            if cancer in text:
                entities['cancer_types'].append(cancer)
                
        for drug in self.drug_lexicon:
            if drug in text:
                entities['drugs'].append(drug)
                
        for marker in self.biomarkers:
            if marker.lower() in text:
                entities['biomarkers'].append(marker)
                
        for gene in self.genes:
            if gene.lower() in text:
                entities['genes'].append(gene)
                
        for mutation in self.mutations:
            if mutation.lower() in text:
                entities['mutations'].append(mutation)
                
        # SpaCy NER extraction
        for ent in doc.ents:
            ent_text = ent.text.lower()
            if ent.label_ in ['CHEMICAL', 'DRUG'] and ent_text not in entities['drugs']:
                entities['other_entities'].append(ent.text)
            elif ent.label_ == 'GENE_OR_GENE_PRODUCT':
                if ent_text not in entities['genes'] and ent_text not in entities['biomarkers']:
                    entities['genes'].append(ent.text)
            elif ent.label_ == 'DISEASE' and ent_text not in entities['cancer_types']:
                entities['other_entities'].append(ent.text)
                
        # Deduplicate and clean
        for key in entities:
            entities[key] = sorted(list(set(entities[key])))
            
        return entities

class RelationshipExtractor:
    """Extracts relationships between entities with improved pattern matching"""

    def __init__(self, processor: TextProcessor):
        self.processor = processor
        self.relationship_patterns = self._init_relationship_patterns()
        logger.info(f"Initialized RelationshipExtractor with {len(self.relationship_patterns)} relationship types")
        
    def _init_relationship_patterns(self) -> Dict[str, List[Tuple[str, re.Pattern]]]:
        """Define compiled regex patterns for different relationship types"""
        patterns = {
            'drug_target': [
                (r'(?P<drug>\w+)\s+(?:targets?|inhibits?|blocks?)\s+(?P<target>\w+)', re.IGNORECASE),
                (r'(?P<target>\w+)\s+(?:is targeted by|is inhibited by)\s+(?P<drug>\w+)', re.IGNORECASE),
                (r'(?:target|inhibition)\s+of\s+(?P<target>\w+)\s+by\s+(?P<drug>\w+)', re.IGNORECASE)
            ],
            'drug_biomarker_response': [
                (r'(?P<biomarker>\w+)[-\s]?(?:positive|status)\s+(?:predicts|correlates with)\s+response to (?P<drug>\w+)', re.IGNORECASE),
                (r'(?P<drug>\w+)\s+(?:response|efficacy)\s+(?:is|was)\s+(?:associated with|predicted by)\s+(?P<biomarker>\w+)', re.IGNORECASE),
                (r'(?P<biomarker>\w+)\s+mutation\s+(?:confers|mediates)\s+resistance to\s+(?P<drug>\w+)', re.IGNORECASE)
            ],
            'drug_synergy': [
                (r'(?P<drug1>\w+)\s+and\s+(?P<drug2>\w+)\s+(?:show|exhibit|demonstrate)\s+synerg', re.IGNORECASE),
                (r'(?:combination|co[\s-]?administration)\s+of\s+(?P<drug1>\w+)\s+and\s+(?P<drug2>\w+)', re.IGNORECASE),
                (r'(?P<drug1>\w+)\s+enhances?\s+the\s+(?:efficacy|effect)\s+of\s+(?P<drug2>\w+)', re.IGNORECASE)
            ],
            'biomarker_cancer': [
                (r'(?P<biomarker>\w+)[-\s]?(?:positive|expression)\s+in\s+(?P<cancer>\w+\s+cancer)', re.IGNORECASE),
                (r'(?P<cancer>\w+\s+cancer)\s+with\s+(?P<biomarker>\w+)\s+(?:mutation|amplification)', re.IGNORECASE),
                (r'(?P<biomarker>\w+)\s+as\s+a\s+biomarker\s+for\s+(?P<cancer>\w+\s+cancer)', re.IGNORECASE)
            ],
            'mutation_prognosis': [
                (r'(?P<mutation>\w+\s+mutation)\s+(?:is associated with|predicts)\s+(?:poor|worse)\s+prognosis', re.IGNORECASE),
                (r'(?P<mutation>\w+)\s+mutation\s+correlates with\s+(?:overall|disease-free)\s+survival', re.IGNORECASE),
                (r'(?:presence of|detection of)\s+(?P<mutation>\w+)\s+(?:mutation|alteration)\s+and\s+clinical\s+outcome', re.IGNORECASE)
            ]
        }
        
        # Compile all patterns
        compiled_patterns = {}
        for rel_type, pattern_list in patterns.items():
            compiled_patterns[rel_type] = []
            for pattern_str, flags in pattern_list:
                try:
                    compiled_pattern = re.compile(pattern_str, flags)
                    compiled_patterns[rel_type].append((pattern_str, compiled_pattern))
                except re.error as e:
                    logger.error(f"Error compiling pattern '{pattern_str}': {str(e)}")
                    continue
                    
        return compiled_patterns
        
    def extract_drug_combinations(self, text: str) -> List[Dict]:
        """
        Specialized extraction of drug combination regimens
        Args:
            text: Input text
        Returns:
            List of combination relationships
        """
        combinations = []
        text_lower = text.lower()
        
        # Pattern 1: "A + B" regimen
        pattern1 = re.compile(
            r'(?P<drug1>\w+)\s*\+\s*(?P<drug2>\w+)\s+(?:regimen|therapy|treatment)',
            re.IGNORECASE
        )
        
        # Pattern 2: "combination of A and B"
        pattern2 = re.compile(
            r'combination (?:therapy )?of (?P<drug1>\w+) and (?P<drug2>\w+)',
            re.IGNORECASE
        )
        
        # Pattern 3: "A/B" notation
        pattern3 = re.compile(
            r'(?P<drug1>\w+)/(?P<drug2>\w+)\s+(?:regimen|therapy)',
            re.IGNORECASE
        )
        
        for pattern in [pattern1, pattern2, pattern3]:
            for match in pattern.finditer(text_lower):
                groups = match.groupdict()
                if len(groups) == 2:
                    drug1, drug2 = groups.values()
                    combinations.append({
                        'type': 'drug_combination_regimen',
                        'entity1': drug1,
                        'entity2': drug2,
                        'evidence': text[match.start():match.end()+100],
                        'source': 'pattern',
                        'score': 0.9
                    })
                    
        return combinations
        
    def extract_biomarker_effects(self, text: str) -> List[Dict]:
        """
        Extract biomarker effects on treatment response
        Args:
            text: Input text
        Returns:
            List of biomarker-effect relationships
        """
        effects = []
        text_lower = text.lower()
        
        patterns = [
            # Biomarker predicts response
            (r'(?P<biomarker>\w+)[-\s]?(?:positive|expression)\s+predicts\s+(?P<effect>response|resistance)\s+to',
             'biomarker_predicts_response'),
            # Biomarker associated with outcome
            (r'(?P<biomarker>\w+)\s+(?:mutation|amplification)\s+associated with\s+(?P<effect>worse|improved)\s+(?:prognosis|outcome)',
             'biomarker_prognostic')
        ]
        
        for pattern_str, rel_type in patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                for match in pattern.finditer(text_lower):
                    groups = match.groupdict()
                    if len(groups) == 2:
                        biomarker, effect = groups.values()
                        effects.append({
                            'type': rel_type,
                            'entity1': biomarker,
                            'entity2': effect,
                            'evidence': text[match.start():match.end()+100],
                            'source': 'pattern',
                            'score': 0.85
                        })
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {pattern_str} - {e}")
                continue
                
        return effects
        
    def extract_relationships(self, text: str, entities: Dict[str, List[str]]) -> List[Dict]:
        """
        Extract relationships from text based on entities and patterns
        Args:
            text: Input text
            entities: Extracted entities from TextProcessor
        Returns:
            List of relationship dictionaries with evidence
        """
        relationships = []
        text_lower = text.lower()
        
        # Co-occurrence relationships (basic)
        self._add_cooccurrence_relationships(relationships, entities, text)
        
        # Pattern-based relationships (advanced)
        self._add_pattern_relationships(relationships, text, text_lower)
        
        # Specialized relationship extractors
        relationships.extend(self.extract_drug_combinations(text))
        relationships.extend(self.extract_biomarker_effects(text))
        
        # Validate relationships against known entities
        valid_relationships = []
        for rel in relationships:
            if self._validate_relationship(rel, entities):
                valid_relationships.append(rel)
                
        logger.debug(f"Extracted {len(valid_relationships)} relationships from text")
        return valid_relationships
        
    def _add_cooccurrence_relationships(self, relationships: List[Dict], entities: Dict[str, List[str]], text: str) -> None:
        """Add co-occurrence based relationships"""
        # Drug-drug relationships
        if len(entities['drugs']) >= 2:
            for i in range(len(entities['drugs'])):
                for j in range(i+1, len(entities['drugs'])):
                    relationships.append({
                        'type': 'drug_drug_cooccurrence',
                        'entity1': entities['drugs'][i],
                        'entity2': entities['drugs'][j],
                        'evidence': f"Co-occurrence in text: {text[:200]}...",
                        'source': 'cooccurrence',
                        'score': 0.5  # Basic confidence score
                    })
                    
        # Drug-gene relationships
        if entities['drugs'] and entities['genes']:
            for drug in entities['drugs']:
                for gene in entities['genes']:
                    relationships.append({
                        'type': 'drug_gene_cooccurrence',
                        'entity1': drug,
                        'entity2': gene,
                        'evidence': f"Co-occurrence in text: {text[:200]}...",
                        'source': 'cooccurrence',
                        'score': 0.6
                    })
                    
        # Biomarker-cancer relationships
        if entities['biomarkers'] and entities['cancer_types']:
            for biomarker in entities['biomarkers']:
                for cancer in entities['cancer_types']:
                    relationships.append({
                        'type': 'biomarker_cancer_cooccurrence',
                        'entity1': biomarker,
                        'entity2': cancer,
                        'evidence': f"Co-occurrence in text: {text[:200]}...",
                        'source': 'cooccurrence',
                        'score': 0.7
                    })

    def _add_pattern_relationships(self, relationships: List[Dict], text: str, text_lower: str) -> None:
        """Add pattern-based relationships using regex"""
        for rel_type, pattern_list in self.relationship_patterns.items():
            for pattern_tuple in pattern_list:
                pattern_name, pattern = pattern_tuple
                # Import required libraries
import requests
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Set
import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import networkx as nx
import re
import time
from urllib.parse import quote
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('oncology_miner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize NLP model
logger.info("Initializing NLP model...")
try:
    nlp = spacy.load("en_core_sci_sm")
    logger.info("Loaded en_core_sci_sm model successfully.")
except OSError:
    logger.warning("Scientific model not found, falling back to medium model...")
    try:
        nlp = spacy.load("en_core_web_md")
        logger.info("Loaded en_core_web_md model successfully.")
    except OSError:
        logger.warning("Medium model not found, falling back to small English model...")
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded en_core_web_sm model successfully.")
        logger.info("Note: For better results, install scispaCy models:")
        logger.info("pip install scispacy")
        logger.info("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz")

# Constants
CACHE_DIR = "./data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
logger.info(f"Using cache directory: {CACHE_DIR}")

# Configure requests session
session = requests.Session()
session.headers.update({'User-Agent': 'OncologyLiteratureMiner/1.0'})

class LiteratureCollector:
    """Handles collection of literature from various sources"""

    def __init__(self):
        self.base_urls = {
            'pubmed': "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            'clinical_trials': "https://clinicaltrials.gov/api/v2/studies",
            'google_patents': "https://patents.googleapis.com/v1/patents:search"
        }
        self.max_retries = 3
        self.retry_delay = 2
        logger.info("LiteratureCollector initialized.")

    def _make_request(self, url: str, params: Dict, is_xml: bool = False):
        for attempt in range(self.max_retries):
            try:
                resp = session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                if is_xml and not resp.content.strip():
                    raise ValueError("Empty XML response")
                return resp
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Request failed: {e}")
                    return None
                time.sleep(self.retry_delay)
    
    def build_pubmed_query(self, drugs: List[str] = None, 
                         biomarkers: List[str] = None,
                         cancers: List[str] = None,
                         genes: List[str] = None,
                         date_range: Tuple[str, str] = None) -> str:
        """
        Build a sophisticated PubMed query from components
        Args:
            drugs: List of drug names
            biomarkers: List of biomarkers
            cancers: List of cancer types
            genes: List of genes
            date_range: Tuple of (start_date, end_date) in YYYY/MM/DD format
        Returns:
            PubMed-compatible query string
        """
        query_parts = []
        
        # Add terms with field tags
        if drugs:
            drug_terms = [f'"{drug}"[Title/Abstract]' for drug in drugs]
            query_parts.append(f"({' OR '.join(drug_terms)})")
            
        if biomarkers:
            bio_terms = [f'"{bm}"[Title/Abstract]' for bm in biomarkers]
            query_parts.append(f"({' OR '.join(bio_terms)})")
            
        if cancers:
            cancer_terms = [f'"{cancer}"[Title/Abstract]' for cancer in cancers]
            query_parts.append(f"({' OR '.join(cancer_terms)})")
            
        if genes:
            gene_terms = [f'"{gene}"[Title/Abstract]' for gene in genes]
            query_parts.append(f"({' OR '.join(gene_terms)})")
            
        # Add date filter if provided
        if date_range:
            start, end = date_range
            query_parts.append(f'("{start}"[Date - Publication] : "{end}"[Date - Publication])')
            
        # Combine with AND
        return ' AND '.join(query_parts) if query_parts else 'cancer'

    def search_pubmed(self, query: str, max_results: int = 1000) -> List[Dict]:
        """
        Search PubMed and return a list of article metadata.
        """
        # 1. Define search parameters up front
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(max_results, 10000),
            'retmode': 'json',
        }
        # Only add the api_key if the env var is set
        api_key = os.getenv('NCBI_API_KEY')
        if api_key:
            search_params['api_key'] = api_key

        # 2. Execute the search
        logger.info(f"Searching PubMed for: '{query}'")
        response = self._make_request(self.base_urls['pubmed'] + 'esearch.fcgi', search_params)
        if not response:
            logger.error("PubMed search request failed")
            return []

        # 3. Parse PMIDs
        data = response.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        if not pmids:
            logger.info("No articles found matching query")
            return []

        logger.info(f"Found {len(pmids)} articles. Fetching details...")

        # 4. Fetch details in batches
        articles = []
        for i in range(0, len(pmids), 100):
            batch = pmids[i:i+100]
            fetch_params = {
                'db': 'pubmed',
                'id': ",".join(batch),
                'retmode': 'xml',
                'api_key': os.getenv('NCBI_API_KEY')
            }
            fetch_resp = self._make_request(self.base_urls['pubmed'] + 'efetch.fcgi',
                                          fetch_params,
                                          is_xml=True)
            if not fetch_resp:
                continue

            # 5. Parse XML
            try:
                root = ET.fromstring(fetch_resp.content)
            except ET.ParseError:
                content = fetch_resp.content
                if content.startswith(b'<?xml'):
                    content = content.split(b'?>', 1)[1]
                root = ET.fromstring(content)

            for art in root.findall('.//PubmedArticle'):
                try:
                    pmid = art.findtext('.//PMID')
                    title = art.findtext('.//ArticleTitle')
                    abstract = ' '.join([t.text for t in art.findall('.//AbstractText') if t.text])
                    journal = art.findtext('.//Journal/Title')
                    year = art.findtext('.//PubDate/Year')
                    month = art.findtext('.//PubDate/Month')
                    day = art.findtext('.//PubDate/Day')
                    pub_date = "-".join(filter(None, [year, month, day]))
                    doi = art.findtext('.//ArticleId[@IdType="doi"]')
                    pmcid = art.findtext('.//ArticleId[@IdType="pmc"]')
                    authors = [f"{a.findtext('LastName')} {a.findtext('ForeName')}"
                            for a in art.findall('.//Author')
                            if a.findtext('LastName') and a.findtext('ForeName')]
                    articles.append({
                        'pmid': pmid,
                        'title': title,
                        'abstract': abstract,
                        'journal': journal,
                        'pub_date': pub_date,
                        'doi': doi,
                        'pmcid': pmcid,
                        'authors': authors
                    })
                except Exception as e:
                    logger.warning(f"Error parsing article XML: {e}")
                    continue

            time.sleep(0.5)

        # 6. Cache and return
        cache_file = os.path.join(CACHE_DIR, f"pubmed_{hash(query)}.json")
        if articles:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(articles, f, indent=2)
                logger.info(f"Cached {len(articles)} articles to {cache_file}")
            except Exception as e:
                logger.error(f"Could not write cache: {e}")

        return articles

    def search_clinical_trials(self, condition: str, intervention: str = None, max_results: int = 100) -> List[Dict]:
        """
        Search ClinicalTrials.gov for oncology trials
        Args:
            condition: Cancer condition to search for
            intervention: Optional intervention filter
            max_results: Maximum number of results to return
        Returns:
            List of trial metadata dictionaries with keys:
            - nct_id: ClinicalTrials.gov identifier
            - title: Brief title
            - official_title: Official title
            - description: Brief summary
            - conditions: List of conditions
            - interventions: List of interventions
            - phase: Trial phase
            - status: Overall status
            - start_date: Start date
            - completion_date: Completion date
            - study_type: Type of study
            - enrollment: Enrollment count
        """
        cache_key = f"clinical_{condition}_{intervention if intervention else ''}"
        cache_file = os.path.join(CACHE_DIR, f"{hash(cache_key)}.json")

        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data:
                        logger.info(f"Loaded {len(cached_data)} trials from cache")
                        return cached_data
            except Exception as e:
                logger.error(f"Error loading cache: {e}")

        # Build query parameters
        params = {
            'query.cond': condition.replace(' ', '+'),
            'format': 'json',
            'pageSize': min(max_results, 1000)
        }
        if intervention:
            params['query.intr'] = intervention.replace(' ', '+')

        # Updated ClinicalTrials.gov API endpoint (v2)
        base_url = "https://clinicaltrials.gov/api/v2/studies"

        logger.info(f"Querying ClinicalTrials.gov for: {condition}")
        if intervention:
            logger.info(f"With intervention: {intervention}")

        try:
            response = session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            processed_trials = []
            for study in data.get('studies', []):
                try:
                    protocol = study.get('protocolSection', {})
                    identification = protocol.get('identificationModule', {})
                    status = protocol.get('statusModule', {})
                    design = protocol.get('designModule', {})
                    description = protocol.get('descriptionModule', {})
                    arms = protocol.get('armsInterventionsModule', {})
                    
                    processed_trial = {
                        'nct_id': identification.get('nctId'),
                        'title': identification.get('briefTitle'),
                        'official_title': identification.get('officialTitle'),
                        'description': description.get('briefSummary'),
                        'conditions': [c.get('name') for c in design.get('conditions', [])],
                        'interventions': [i.get('name') for i in arms.get('interventions', [])],
                        'phase': design.get('phase'),
                        'status': status.get('overallStatus'),
                        'start_date': status.get('startDate'),
                        'completion_date': status.get('completionDate'),
                        'study_type': design.get('studyType'),
                        'enrollment': design.get('enrollmentInfo', {}).get('count')
                    }
                    processed_trials.append(processed_trial)
                except Exception as e:
                    logger.warning(f"Error processing trial: {str(e)}")
                    continue

            # Cache results
            if processed_trials:
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(processed_trials, f, indent=2)
                    logger.info(f"Cached {len(processed_trials)} trials")
                except Exception as e:
                    logger.error(f"Could not cache results: {str(e)}")

            return processed_trials

        except Exception as e:
            logger.error(f"Error accessing ClinicalTrials.gov API: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"API response: {e.response.text[:500]}")
            return []

    def search_google_patents(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search Google Patents for oncology-related patents
        Args:
            query: Search query
            max_results: Maximum number of results
        Returns:
            List of patent dictionaries
        """
        cache_key = f"patents_{query}"
        cache_file = os.path.join(CACHE_DIR, f"{hash(cache_key)}.json")
        
        # Check cache first
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading patent cache: {e}")
                
        params = {
            'query': query,
            'pageSize': min(max_results, 100),
            'key': os.getenv('GOOGLE_PATENTS_API_KEY', '')  # Add your API key
        }
        
        try:
            logger.info(f"Searching Google Patents for: {query}")
            response = session.get(self.base_urls['google_patents'], params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            patents = []
            for patent in data.get('patents', []):
                try:
                    patents.append({
                        'patent_id': patent.get('patentId'),
                        'title': patent.get('title'),
                        'abstract': patent.get('abstractText'),
                        'filing_date': patent.get('filingDate'),
                        'publication_date': patent.get('publicationDate'),
                        'inventors': [inv.get('name') for inv in patent.get('inventors', [])],
                        'assignees': [ass.get('name') for ass in patent.get('assignees', [])],
                        'claims': [claim.get('text') for claim in patent.get('claims', [])]
                    })
                except Exception as e:
                    logger.warning(f"Error processing patent: {e}")
                    continue
                    
            # Cache results
            if patents:
                with open(cache_file, 'w') as f:
                    json.dump(patents, f, indent=2)
                    
            return patents
            
        except Exception as e:
            logger.error(f"Error accessing Google Patents API: {e}")
            return []

class TextProcessor:
    """Handles text preprocessing and entity recognition"""

    def __init__(self):
        logger.info("Initializing TextProcessor with oncology dictionaries...")
        self.cancer_types = self._load_cancer_types()
        self.drug_lexicon = self._load_drug_lexicon()
        self.biomarkers = self._load_biomarkers()
        self.genes = self._load_genes()
        self.mutations = self._load_mutations()
        
        logger.info(f"Loaded {len(self.cancer_types)} cancer types, {len(self.drug_lexicon)} drugs, "
                   f"{len(self.biomarkers)} biomarkers, {len(self.genes)} genes, "
                   f"{len(self.mutations)} mutations")
        
    def _load_cancer_types(self) -> Set[str]:
        """Load cancer types from MeSH/ICD-10"""
        return {
            'breast cancer', 'lung cancer', 'colorectal cancer', 'prostate cancer',
            'melanoma', 'leukemia', 'lymphoma', 'glioblastoma', 'pancreatic cancer',
            'ovarian cancer', 'bladder cancer', 'hepatocellular carcinoma',
            'non-small cell lung cancer', 'triple-negative breast cancer',
            'renal cell carcinoma', 'gastric cancer', 'esophageal cancer',
            'head and neck cancer', 'cervical cancer', 'endometrial cancer'
        }
        
    def _load_drug_lexicon(self) -> Dict[str, str]:
        """Load drug names and categories"""
        return {
            'trastuzumab': 'targeted therapy', 'pertuzumab': 'targeted therapy',
            'pembrolizumab': 'immunotherapy', 'nivolumab': 'immunotherapy',
            'ipilimumab': 'immunotherapy', 'atezolizumab': 'immunotherapy',
            'cisplatin': 'chemotherapy', 'carboplatin': 'chemotherapy',
            'paclitaxel': 'chemotherapy', 'docetaxel': 'chemotherapy',
            'doxorubicin': 'chemotherapy', 'cyclophosphamide': 'chemotherapy',
            'tamoxifen': 'hormone therapy', 'letrozole': 'hormone therapy',
            'olaparib': 'targeted therapy', 'rucaparib': 'targeted therapy',
            'erlotinib': 'targeted therapy', 'gefitinib': 'targeted therapy',
            'osimertinib': 'targeted therapy', 'cetuximab': 'targeted therapy'
        }
        
    def _load_biomarkers(self) -> Set[str]:
        """Load known biomarkers"""
        return {
            'PD-L1', 'HER2', 'ER', 'PR', 'KRAS', 'NRAS', 'BRCA1', 'BRCA2',
            'EGFR', 'ALK', 'BRAF', 'MSI', 'TMB', 'TP53', 'PTEN', 'PIK3CA',
            'ROS1', 'MET', 'NTRK', 'FGFR', 'RET', 'AR', 'CDK4', 'CDK6'
        }
        
    def _load_genes(self) -> Set[str]:
        """Load cancer-related genes"""
        return {
            'EGFR', 'HER2', 'KRAS', 'NRAS', 'BRAF', 'ALK', 'ROS1', 'MET',
            'BRCA1', 'BRCA2', 'TP53', 'PTEN', 'PIK3CA', 'MYC', 'CDKN2A',
            'RB1', 'APC', 'NOTCH1', 'JAK2', 'STAT3', 'NF1', 'NF2', 'VHL'
        }
        
    def _load_mutations(self) -> Set[str]:
        """Load common cancer mutations"""
        return {
            'EGFR L858R', 'EGFR T790M', 'KRAS G12D', 'KRAS G12V',
            'BRAF V600E', 'PIK3CA H1047R', 'TP53 R175H', 'BRCA1 5382insC',
            'ALK EML4-ALK', 'HER2 amplification', 'MET exon 14 skipping'
        }
        
    def clean_text(self, text: str) -> str:
        """
        Advanced text cleaning pipeline
        Args:
            text: Input text
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove references/citations
        text = re.sub(r'\[[\d,]+\]', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)
        
        return text
        
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text:
            return ""
            
        # Remove special characters, extra whitespace, and normalize
        text = re.sub(r'[^\w\s-]', ' ', text.lower())  # Keep words, spaces, and hyphens
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def normalize_entity(self, entity: str, entity_type: str) -> str:
        """
        Normalize entity names to standard forms
        Args:
            entity: Entity text
            entity_type: Type of entity
        Returns:
            Normalized entity name
        """
        if not entity:
            return ""
            
        # Common normalizations
        entity = entity.strip().lower()
        
        # Type-specific normalizations
        if entity_type == 'drug':
            # Remove trade names in parentheses
            entity = re.sub(r'\(.*?\)', '', entity)
            # Standardize endings
            if entity.endswith('mab'):
                entity = entity[:-3] + 'mab'  # Standardize monoclonal antibodies
            elif entity.endswith('nib'):
                entity = entity[:-3] + 'nib'  # Standardize kinase inhibitors
                
        elif entity_type in ['gene', 'biomarker']:
            # Standardize gene names
            entity = entity.upper()
            # Remove common prefixes
            entity = re.sub(r'^(gene|protein)\s+', '', entity)
            
        elif entity_type == 'cancer_type':
            # Standardize cancer names
            if not entity.endswith(' cancer'):
                entity = f"{entity} cancer"
                
        return entity.strip()
        
    def link_entities_to_databases(self, entities: Dict[str, List[str]]) -> Dict[str, List[Dict]]:
        """
        Link entities to standard databases
        Args:
            entities: Dictionary of entity lists
        Returns:
            Dictionary with linked entities including database IDs
        """
        linked_entities = {k: [] for k in entities.keys()}
        
        # Mock database links - in practice you'd use APIs or local databases
        db_mappings = {
            'drugs': {
                'trastuzumab': {'drugbank': 'DB00072', 'chebi': 'CHEBI:72447'},
                'pembrolizumab': {'drugbank': 'DB09037', 'chebi': 'CHEBI:90551'}
            },
            'genes': {
                'BRCA1': {'entrez': '672', 'hgnc': '1100'},
                'EGFR': {'entrez': '1956', 'hgnc': '3236'}
            },
            'cancer_types': {
                'breast cancer': {'mesh': 'D001943', 'icd10': 'C50'},
                'lung cancer': {'mesh': 'D002283', 'icd10': 'C34'}
            }
        }
        
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                norm_entity = self.normalize_entity(entity, entity_type)
                db_links = db_mappings.get(entity_type, {}).get(norm_entity, {})
                
                linked_entities[entity_type].append({
                    'text': entity,
                    'normalized': norm_entity,
                    'db_links': db_links
                })
                
        return linked_entities
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract biomedical entities from text
        Args:
            text: Input text to process
        Returns:
            Dictionary of entity types and their values
        """
        if not text:
            return {}
            
        text = self.preprocess_text(text)
        doc = nlp(text)
        
        entities = {
            'cancer_types': [],
            'drugs': [],
            'biomarkers': [],
            'genes': [],
            'mutations': [],
            'other_entities': []
        }
        
        # Rule-based extraction
        for cancer in self.cancer_types:
            if cancer in text:
                entities['cancer_types'].append(cancer)
                
        for drug in self.drug_lexicon:
            if drug in text:
                entities['drugs'].append(drug)
                
        for marker in self.biomarkers:
            if marker.lower() in text:
                entities['biomarkers'].append(marker)
                
        for gene in self.genes:
            if gene.lower() in text:
                entities['genes'].append(gene)
                
        for mutation in self.mutations:
            if mutation.lower() in text:
                entities['mutations'].append(mutation)
                
        # SpaCy NER extraction
        for ent in doc.ents:
            ent_text = ent.text.lower()
            if ent.label_ in ['CHEMICAL', 'DRUG'] and ent_text not in entities['drugs']:
                entities['other_entities'].append(ent.text)
            elif ent.label_ == 'GENE_OR_GENE_PRODUCT':
                if ent_text not in entities['genes'] and ent_text not in entities['biomarkers']:
                    entities['genes'].append(ent.text)
            elif ent.label_ == 'DISEASE' and ent_text not in entities['cancer_types']:
                entities['other_entities'].append(ent.text)
                
        # Deduplicate and clean
        for key in entities:
            entities[key] = sorted(list(set(entities[key])))
            
        return entities

class RelationshipExtractor:
    """Extracts relationships between entities with improved pattern matching"""

    def __init__(self, processor: TextProcessor):
        self.processor = processor
        self.relationship_patterns = self._init_relationship_patterns()
        logger.info(f"Initialized RelationshipExtractor with {len(self.relationship_patterns)} relationship types")
        
    def _init_relationship_patterns(self) -> Dict[str, List[Tuple[str, re.Pattern]]]:
        """Define compiled regex patterns for different relationship types"""
        patterns = {
            'drug_target': [
                (r'(?P<drug>\w+)\s+(?:targets?|inhibits?|blocks?)\s+(?P<target>\w+)', re.IGNORECASE),
                (r'(?P<target>\w+)\s+(?:is targeted by|is inhibited by)\s+(?P<drug>\w+)', re.IGNORECASE),
                (r'(?:target|inhibition)\s+of\s+(?P<target>\w+)\s+by\s+(?P<drug>\w+)', re.IGNORECASE)
            ],
            'drug_biomarker_response': [
                (r'(?P<biomarker>\w+)[-\s]?(?:positive|status)\s+(?:predicts|correlates with)\s+response to (?P<drug>\w+)', re.IGNORECASE),
                (r'(?P<drug>\w+)\s+(?:response|efficacy)\s+(?:is|was)\s+(?:associated with|predicted by)\s+(?P<biomarker>\w+)', re.IGNORECASE),
                (r'(?P<biomarker>\w+)\s+mutation\s+(?:confers|mediates)\s+resistance to\s+(?P<drug>\w+)', re.IGNORECASE)
            ],
            'drug_synergy': [
                (r'(?P<drug1>\w+)\s+and\s+(?P<drug2>\w+)\s+(?:show|exhibit|demonstrate)\s+synerg', re.IGNORECASE),
                (r'(?:combination|co[\s-]?administration)\s+of\s+(?P<drug1>\w+)\s+and\s+(?P<drug2>\w+)', re.IGNORECASE),
                (r'(?P<drug1>\w+)\s+enhances?\s+the\s+(?:efficacy|effect)\s+of\s+(?P<drug2>\w+)', re.IGNORECASE)
            ],
            'biomarker_cancer': [
                (r'(?P<biomarker>\w+)[-\s]?(?:positive|expression)\s+in\s+(?P<cancer>\w+\s+cancer)', re.IGNORECASE),
                (r'(?P<cancer>\w+\s+cancer)\s+with\s+(?P<biomarker>\w+)\s+(?:mutation|amplification)', re.IGNORECASE),
                (r'(?P<biomarker>\w+)\s+as\s+a\s+biomarker\s+for\s+(?P<cancer>\w+\s+cancer)', re.IGNORECASE)
            ],
            'mutation_prognosis': [
                (r'(?P<mutation>\w+\s+mutation)\s+(?:is associated with|predicts)\s+(?:poor|worse)\s+prognosis', re.IGNORECASE),
                (r'(?P<mutation>\w+)\s+mutation\s+correlates with\s+(?:overall|disease-free)\s+survival', re.IGNORECASE),
                (r'(?:presence of|detection of)\s+(?P<mutation>\w+)\s+(?:mutation|alteration)\s+and\s+clinical\s+outcome', re.IGNORECASE)
            ]
        }
        
        # Compile all patterns
        compiled_patterns = {}
        for rel_type, pattern_list in patterns.items():
            compiled_patterns[rel_type] = []
            for pattern_str, flags in pattern_list:
                try:
                    compiled_pattern = re.compile(pattern_str, flags)
                    compiled_patterns[rel_type].append((pattern_str, compiled_pattern))
                except re.error as e:
                    logger.error(f"Error compiling pattern '{pattern_str}': {str(e)}")
                    continue
                    
        return compiled_patterns
        
    def extract_drug_combinations(self, text: str) -> List[Dict]:
        """
        Specialized extraction of drug combination regimens
        Args:
            text: Input text
        Returns:
            List of combination relationships
        """
        combinations = []
        text_lower = text.lower()
        
        # Pattern 1: "A + B" regimen
        pattern1 = re.compile(
            r'(?P<drug1>\w+)\s*\+\s*(?P<drug2>\w+)\s+(?:regimen|therapy|treatment)',
            re.IGNORECASE
        )
        
        # Pattern 2: "combination of A and B"
        pattern2 = re.compile(
            r'combination (?:therapy )?of (?P<drug1>\w+) and (?P<drug2>\w+)',
            re.IGNORECASE
        )
        
        # Pattern 3: "A/B" notation
        pattern3 = re.compile(
            r'(?P<drug1>\w+)/(?P<drug2>\w+)\s+(?:regimen|therapy)',
            re.IGNORECASE
        )
        
        for pattern in [pattern1, pattern2, pattern3]:
            for match in pattern.finditer(text_lower):
                groups = match.groupdict()
                if len(groups) == 2:
                    drug1, drug2 = groups.values()
                    combinations.append({
                        'type': 'drug_combination_regimen',
                        'entity1': drug1,
                        'entity2': drug2,
                        'evidence': text[match.start():match.end()+100],
                        'source': 'pattern',
                        'score': 0.9
                    })
                    
        return combinations
        
    def extract_biomarker_effects(self, text: str) -> List[Dict]:
        """
        Extract biomarker effects on treatment response
        Args:
            text: Input text
        Returns:
            List of biomarker-effect relationships
        """
        effects = []
        text_lower = text.lower()
        
        patterns = [
            # Biomarker predicts response
            (r'(?P<biomarker>\w+)[-\s]?(?:positive|expression)\s+predicts\s+(?P<effect>response|resistance)\s+to',
             'biomarker_predicts_response'),
            # Biomarker associated with outcome
            (r'(?P<biomarker>\w+)\s+(?:mutation|amplification)\s+associated with\s+(?P<effect>worse|improved)\s+(?:prognosis|outcome)',
             'biomarker_prognostic')
        ]
        
        for pattern_str, rel_type in patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                for match in pattern.finditer(text_lower):
                    groups = match.groupdict()
                    if len(groups) == 2:
                        biomarker, effect = groups.values()
                        effects.append({
                            'type': rel_type,
                            'entity1': biomarker,
                            'entity2': effect,
                            'evidence': text[match.start():match.end()+100],
                            'source': 'pattern',
                            'score': 0.85
                        })
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {pattern_str} - {e}")
                continue
                
        return effects
        
    def extract_relationships(self, text: str, entities: Dict[str, List[str]]) -> List[Dict]:
        """
        Extract relationships from text based on entities and patterns
        Args:
            text: Input text
            entities: Extracted entities from TextProcessor
        Returns:
            List of relationship dictionaries with evidence
        """
        relationships = []
        text_lower = text.lower()
        
        # Co-occurrence relationships (basic)
        self._add_cooccurrence_relationships(relationships, entities, text)
        
        # Pattern-based relationships (advanced)
        self._add_pattern_relationships(relationships, text, text_lower)
        
        # Specialized relationship extractors
        relationships.extend(self.extract_drug_combinations(text))
        relationships.extend(self.extract_biomarker_effects(text))
        
        # Validate relationships against known entities
        valid_relationships = []
        for rel in relationships:
            if self._validate_relationship(rel, entities):
                valid_relationships.append(rel)
                
        logger.debug(f"Extracted {len(valid_relationships)} relationships from text")
        return valid_relationships
        
    def _add_cooccurrence_relationships(self, relationships: List[Dict], entities: Dict[str, List[str]], text: str) -> None:
        """Add co-occurrence based relationships"""
        # Drug-drug relationships
        if len(entities['drugs']) >= 2:
            for i in range(len(entities['drugs'])):
                for j in range(i+1, len(entities['drugs'])):
                    relationships.append({
                        'type': 'drug_drug_cooccurrence',
                        'entity1': entities['drugs'][i],
                        'entity2': entities['drugs'][j],
                        'evidence': f"Co-occurrence in text: {text[:200]}...",
                        'source': 'cooccurrence',
                        'score': 0.5  # Basic confidence score
                    })
                    
        # Drug-gene relationships
        if entities['drugs'] and entities['genes']:
            for drug in entities['drugs']:
                for gene in entities['genes']:
                    relationships.append({
                        'type': 'drug_gene_cooccurrence',
                        'entity1': drug,
                        'entity2': gene,
                        'evidence': f"Co-occurrence in text: {text[:200]}...",
                        'source': 'cooccurrence',
                        'score': 0.6
                    })
                    
        # Biomarker-cancer relationships
        if entities['biomarkers'] and entities['cancer_types']:
            for biomarker in entities['biomarkers']:
                for cancer in entities['cancer_types']:
                    relationships.append({
                        'type': 'biomarker_cancer_cooccurrence',
                        'entity1': biomarker,
                        'entity2': cancer,
                        'evidence': f"Co-occurrence in text: {text[:200]}...",
                        'source': 'cooccurrence',
                        'score': 0.7
                    })

    def _add_pattern_relationships(self, relationships: List[Dict], text: str, text_lower: str) -> None:
        """Add pattern-based relationships using regex"""
        for rel_type, pattern_list in self.relationship_patterns.items():
            for pattern_tuple in pattern_list:
                pattern_name, pattern = pattern_tuple
                for match in pattern.finditer(text_lower):
                    groups = match.groupdict()
                    if len(groups) == 2:  # All our patterns have exactly 2 named groups
                        entity1, entity2 = groups.values()
                        relationships.append({
                            'type': rel_type,
                            'entity1': entity1,
                            'entity2': entity2,
                            'evidence': f"Pattern match ({pattern_name}): {text[match.start():match.end()+50]}...",
                            'source': 'pattern',
                            'score': 0.8  # Higher confidence for pattern matches
                        })

    def _validate_relationship(self, relationship: Dict, entities: Dict[str, List[str]]) -> bool:
        """Validate that relationship entities exist in our extracted entities"""
        ent1 = relationship['entity1'].lower()
        ent2 = relationship['entity2'].lower()
        
        # Check if entities exist in any of our categories
        ent1_valid = any(ent1 in [e.lower() for e in entities[cat]] 
                        for cat in ['drugs', 'genes', 'biomarkers', 'cancer_types', 'mutations'])
        ent2_valid = any(ent2 in [e.lower() for e in entities[cat]] 
                        for cat in ['drugs', 'genes', 'biomarkers', 'cancer_types', 'mutations'])
        
        if not (ent1_valid and ent2_valid):
            logger.debug(f"Invalid relationship - entities not found: {relationship}")
            return False
            
        # Additional validation based on relationship type
        rel_type = relationship['type']
        
        if 'drug' in rel_type:
            drugs = [e.lower() for e in entities['drugs']]
            if 'drug_drug' in rel_type:
                return ent1 in drugs and ent2 in drugs
            elif 'drug_gene' in rel_type or 'drug_target' in rel_type:
                genes = [e.lower() for e in entities['genes']]
                return (ent1 in drugs and ent2 in genes) or (ent2 in drugs and ent1 in genes)
            elif 'drug_biomarker' in rel_type:
                biomarkers = [e.lower() for e in entities['biomarkers']]
                return (ent1 in drugs and ent2 in biomarkers) or (ent2 in drugs and ent1 in biomarkers)
                
        elif 'biomarker_cancer' in rel_type:
            cancers = [e.lower() for e in entities['cancer_types']]
            biomarkers = [e.lower() for e in entities['biomarkers']]
            return (ent1 in biomarkers and ent2 in cancers) or (ent2 in biomarkers and ent1 in cancers)
            
        return True

class KnowledgeGraph:
    """Efficient implementation of an oncology knowledge graph"""
    
    def __init__(self, processor):
        self.nodes: Set[str] = set()
        self.edges: List[Dict] = []
        self.node_attributes: Dict[str, Dict] = {}
        self.edge_lookup: Dict[Union[Tuple, FrozenSet], bool] = {}  # For fast edge existence checks
        self.processor = processor
        self.node_counter = 0
        self.edge_counter = 0
        logger.info("Initialized empty KnowledgeGraph")
        
    def add_relationship(self, relationship: Dict) -> None:
        """
        Add a relationship to the knowledge graph with validation
        Args:
            relationship: Dictionary containing relationship details
        """
        entity1 = relationship['entity1']
        entity2 = relationship['entity2']
        rel_type = relationship['type']
        
        # Add nodes with attributes if they don't exist
        for entity in (entity1, entity2):
            if entity not in self.nodes:
                self.nodes.add(entity)
                self._set_node_attributes(entity)
                
        # Create edge key (undirected for co-occurrence, directed for others)
        if 'cooccurrence' in rel_type:
            edge_key = frozenset({entity1, entity2, rel_type})
        else:
            edge_key = (entity1, entity2, rel_type)
            
        # Fast existence check using edge_lookup
        if edge_key in self.edge_lookup:
            logger.debug(f"Relationship already exists: {entity1} -- {rel_type} -> {entity2}")
            return
            
        # Add the edge
        edge_id = f"e{self.edge_counter}"
        self.edge_counter += 1
        
        edge_data = {
            'source': entity1,
            'target': entity2,
            'type': rel_type,
            'evidence': relationship.get('evidence', ''),
            'source_type': relationship.get('source', 'unknown'),
            'score': float(relationship.get('score', 0.5)),
            'id': edge_id
        }
        
        self.edges.append(edge_data)
        self.edge_lookup[edge_key] = True  # Mark edge as existing
        logger.debug(f"Added relationship: {entity1} -- {rel_type} -> {entity2}")
        
    def _set_node_attributes(self, node: str) -> None:
        """Efficiently set attributes for a node based on its type"""
        node_lower = node.lower()
        attrs = {'id': f"n{self.node_counter}", 'label': node}
        self.node_counter += 1
        
        # Pre-compute type checks
        is_drug = node_lower in self.processor.drug_lexicon
        is_biomarker = node in self.processor.biomarkers
        is_gene = node in self.processor.genes
        is_cancer = node_lower in self.processor.cancer_types
        is_mutation = any(mut.lower() in node_lower for mut in self.processor.mutations)
        
        # Determine node type and attributes
        if is_drug:
            attrs.update({
                'type': 'drug',
                'category': self.processor.drug_lexicon.get(node_lower, 'unknown'),
                'color': '#1f77b4'  # Blue
            })
        elif is_biomarker:
            attrs.update({
                'type': 'biomarker',
                'color': '#2ca02c'  # Green
            })
        elif is_gene:
            attrs.update({
                'type': 'gene',
                'color': '#ff7f0e'  # Orange
            })
        elif is_cancer:
            attrs.update({
                'type': 'cancer_type',
                'color': '#d62728'  # Red
            })
        elif is_mutation:
            attrs.update({
                'type': 'mutation',
                'color': '#9467bd'  # Purple
            })
        else:
            attrs.update({
                'type': 'other',
                'color': '#7f7f7f'  # Gray
            })
            
        self.node_attributes[node] = attrs
        
    def to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert knowledge graph to pandas DataFrames more efficiently
        Returns:
            Tuple of (nodes_df, edges_df)
        """
        # Nodes DataFrame using list comprehension
        nodes_data = [{
            'id': attrs.get('id'),
            'label': node,
            'type': attrs.get('type', 'unknown'),
            'category': attrs.get('category', ''),
            'color': attrs.get('color', '#999999')
        } for node, attrs in self.node_attributes.items()]
        
        # Edges DataFrame
        edges_df = pd.DataFrame(self.edges)
        
        return pd.DataFrame(nodes_data), edges_df
        
    def to_networkx(self) -> nx.Graph:
        """
        Optimized conversion to NetworkX graph object
        Returns:
            NetworkX graph
        """
        # Use appropriate graph type based on edges
        use_digraph = len(self.edges) <= 1000 and any(
            'cooccurrence' not in e['type'] for e in self.edges
        )
        G = nx.DiGraph() if use_digraph else nx.Graph()
        
        # Batch add nodes with attributes
        G.add_nodes_from((node, attrs) for node, attrs in self.node_attributes.items())
            
        # Batch add edges with attributes
        edge_data = [
            (e['source'], e['target'], {k: v for k, v in e.items() if k not in ['source', 'target']})
            for e in self.edges
        ]
        G.add_edges_from(edge_data)
            
        return G
        
    def analyze_network(self) -> Dict:
        """
        Optimized network analysis with early termination for large graphs
        Returns:
            Dictionary of network metrics
        """
        G = self.to_networkx()
        num_nodes = len(G.nodes())
        num_edges = len(G.edges())
        
        metrics = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': nx.density(G),
            'connected_components': nx.number_connected_components(G) if not nx.is_directed(G) else None,
        }
        
        # Skip expensive computations for very large graphs
        if num_nodes > 1000:
            metrics['note'] = 'Centrality measures skipped for large graph'
            return metrics
            
        # Only compute these for smaller graphs
        try:
            metrics.update({
                'strongly_connected_components': nx.number_strongly_connected_components(G) if nx.is_directed(G) else None,
                'average_clustering': nx.average_clustering(G),
                'degree_assortativity': nx.degree_assortativity_coefficient(G),
                'degree_centrality': dict(sorted(
                    nx.degree_centrality(G).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]),
                'betweenness_centrality': dict(sorted(
                    nx.betweenness_centrality(G, k=min(100, num_nodes)).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
            })
        except Exception as e:
            logger.warning(f"Could not compute some metrics: {e}")
                
        return metrics
        
    def save_gexf(self, filename: str = "oncology_graph.gexf") -> None:
        """
        Save graph to GEXF format with error handling
        Args:
            filename: Output filename
        """
        try:
            G = self.to_networkx()
            os.makedirs("./output", exist_ok=True)
            output_path = os.path.join("./output", filename)
            
            # Use faster write method for large graphs
            if len(G.nodes()) > 5000:
                nx.write_gexf(G, output_path, prettyprint=False)
            else:
                nx.write_gexf(G, output_path)
                
            logger.info(f"Graph saved to GEXF format: {output_path}")
        except Exception as e:
            logger.error(f"Error saving GEXF file: {e}")
            raise

    def visualize(self, max_nodes: int = 50, figsize: Tuple[int, int] = (15, 15)) -> None:
        """
        Optimized visualization with better filtering
        Args:
            max_nodes: Maximum number of nodes to display
            figsize: Figure size (width, height)
        """
        if len(self.nodes) == 0:
            logger.warning("No nodes to visualize")
            return
            
        # Early filtering for large graphs
        if len(self.nodes) > max_nodes:
            logger.info(f"Filtering graph from {len(self.nodes)} to {max_nodes} nodes")
            nodes_df, edges_df = self._get_filtered_subgraph(max_nodes)
        else:
            nodes_df, edges_df = self.to_dataframe()
        
        # Create graph from filtered data
        graph = nx.from_pandas_edgelist(
            edges_df,
            source='source',
            target='target',
            edge_attr=True,
            create_using=nx.DiGraph() if len(edges_df) <= 1000 else nx.Graph()
        )
        
        # Add node attributes
        node_attrs = nodes_df.set_index('label').to_dict('index')
        nx.set_node_attributes(graph, node_attrs)
        
        # Visualization setup
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
        
        # Draw nodes
        node_colors = [data['color'] for _, data in graph.nodes(data=True)]
        node_sizes = [800 + 50 * degree for _, degree in graph.degree()]
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
        
        # Draw edges with styles
        edge_styles = {
            'drug_target': 'solid',
            'drug_biomarker_response': 'dashed',
            'drug_synergy': 'dotted',
            'biomarker_cancer': 'solid',
            'mutation_prognosis': 'dashed',
            'default': 'solid'
        }
        
        for edge_type in edges_df['type'].unique():
            edge_subset = [(u, v) for u, v, e in graph.edges(data=True) if e['type'] == edge_type]
            style = edge_styles.get(edge_type, edge_styles['default'])
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=edge_subset,
                width=1.5,
                edge_color='#777777',
                style=style,
                alpha=0.7
            )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, pos,
            labels={n: n for n in graph.nodes()},
            font_size=8,
            font_family='sans-serif',
            alpha=0.9
        )
        
        # Create legend
        unique_types = nodes_df['type'].unique()
        legend_elements = [
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=typ,
                markerfacecolor=nodes_df[nodes_df['type'] == typ]['color'].iloc[0],
                markersize=10
            ) for typ in unique_types
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.title("Oncology Knowledge Graph", pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def _get_filtered_subgraph(self, max_nodes: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Helper method to filter the graph to most important nodes
        Args:
            max_nodes: Maximum number of nodes to keep
        Returns:
            Filtered nodes and edges DataFrames
        """
        # Get node degrees more efficiently
        source_counts = defaultdict(int)
        target_counts = defaultdict(int)
        
        for edge in self.edges:
            source_counts[edge['source']] += 1
            target_counts[edge['target']] += 1
            
        node_degrees = {
            node: source_counts.get(node, 0) + target_counts.get(node, 0)
            for node in self.nodes
        }
        
        # Get top nodes
        top_nodes = sorted(node_degrees.keys(), key=lambda x: -node_degrees[x])[:max_nodes]
        top_nodes_set = set(top_nodes)
        
        # Filter nodes and edges
        nodes_df = pd.DataFrame([
            self.node_attributes[node] for node in top_nodes
        ])
        
        edges_df = pd.DataFrame([
            edge for edge in self.edges
            if edge['source'] in top_nodes_set and edge['target'] in top_nodes_set
        ])
        
        return nodes_df, edges_df
        
    def to_networkx(self) -> nx.Graph:
        """
        Convert to NetworkX graph object
        Returns:
            NetworkX graph
        """
        G = nx.Graph() if len(self.edges) > 1000 else nx.DiGraph()
        
        # Add nodes with attributes
        for node in self.nodes:
            attrs = self.node_attributes.get(node, {})
            G.add_node(node, **attrs)
            
        # Add edges with attributes
        for edge in self.edges:
            edge_attrs = {k: v for k, v in edge.items() 
                         if k not in ['source', 'target']}
            G.add_edge(edge['source'], edge['target'], **edge_attrs)
            
        return G
        
    def analyze_network(self) -> Dict:
        """
        Analyze network properties
        Returns:
            Dictionary of network metrics
        """
        G = self.to_networkx()
        
        metrics = {
            'num_nodes': len(G.nodes()),
            'num_edges': len(G.edges()),
            'density': nx.density(G),
            'connected_components': nx.number_connected_components(G) if not nx.is_directed(G) else None,
            'strongly_connected_components': nx.number_strongly_connected_components(G) if nx.is_directed(G) else None,
            'average_clustering': nx.average_clustering(G),
            'degree_assortativity': nx.degree_assortativity_coefficient(G)
        }
        
        # Centrality measures for top nodes
        if len(G.nodes()) <= 1000:  # Avoid expensive computations for large graphs
            try:
                metrics['degree_centrality'] = dict(sorted(
                    nx.degree_centrality(G).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
                
                metrics['betweenness_centrality'] = dict(sorted(
                    nx.betweenness_centrality(G).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])
            except Exception as e:
                logger.warning(f"Could not compute centrality measures: {e}")
                
        return metrics
        
    def save_gexf(self, filename: str = "oncology_graph.gexf") -> None:
        """
        Save graph to GEXF format for visualization in Gephi
        Args:
            filename: Output filename
        """
        try:
            G = self.to_networkx()
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            nx.write_gexf(G, output_path)
            logger.info(f"Graph saved to GEXF format: {output_path}")
        except Exception as e:
            logger.error(f"Error saving GEXF file: {e}")

    def visualize(self, max_nodes: int = 50, figsize: Tuple[int, int] = (15, 15)) -> None:
        """
        Visualize a subset of the knowledge graph
        Args:
            max_nodes: Maximum number of nodes to display
            figsize: Figure size (width, height)
        """
        nodes_df, edges_df = self.to_dataframe()
        
        if len(nodes_df) > max_nodes:
            logger.info(f"Graph too large ({len(nodes_df)} nodes), filtering to top {max_nodes} connected nodes...")
            # Get most connected nodes
            node_degrees = pd.concat([
                edges_df['source'].value_counts(),
                edges_df['target'].value_counts()
            ]).groupby(level=0).sum()
            
            top_nodes = node_degrees.nlargest(max_nodes).index.tolist()
            nodes_df = nodes_df[nodes_df['label'].isin(top_nodes)]
            edges_df = edges_df[
                edges_df['source'].isin(top_nodes) & 
                edges_df['target'].isin(top_nodes)
            ]
            
        # Create graph
        plt.figure(figsize=figsize)
        graph = nx.from_pandas_edgelist(
            edges_df,
            source='source',
            target='target',
            edge_attr=True
        )
        
        # Add node attributes
        for _, row in nodes_df.iterrows():
            if row['label'] in graph.nodes():
                graph.nodes[row['label']].update(row.to_dict())
        
        # Layout and drawing
        pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
        
        # Draw nodes with colors
        node_colors = [data['color'] for _, data in graph.nodes(data=True)]
        node_sizes = [800 + 50 * degree for _, degree in graph.degree()]
        
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9
        )
        
        # Draw edges with different styles based on type
        edge_styles = {
            'drug_target': 'solid',
            'drug_biomarker_response': 'dashed',
            'drug_synergy': 'dotted',
            'biomarker_cancer': 'solid',
            'mutation_prognosis': 'dashed',
            'default': 'solid'
        }
        
        for edge_type in edges_df['type'].unique():
            edge_subset = [(u, v) for u, v, e in graph.edges(data=True) if e['type'] == edge_type]
            style = edge_styles.get(edge_type, edge_styles['default'])
            nx.draw_networkx_edges(
                graph, pos,
                edgelist=edge_subset,
                width=1.5,
                edge_color='#777777',
                style=style,
                alpha=0.7
            )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, pos,
            labels={n: n for n in graph.nodes()},
            font_size=8,
            font_family='sans-serif',
            alpha=0.9
        )
        
        # Create legend
        legend_elements = []
        for node_type in nodes_df['type'].unique():
            color = nodes_df[nodes_df['type'] == node_type]['color'].iloc[0]
            legend_elements.append(plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=node_type,
                markerfacecolor=color,
                markersize=10
            ))
            
        plt.legend(
            handles=legend_elements,
            loc='upper right',
            bbox_to_anchor=(1.1, 1.1))
        
        plt.title("Oncology Knowledge Graph", pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

class OncologyLiteratureMiner:
    """End-to-end literature mining pipeline"""

    def __init__(self):
        logger.info("\n" + "="*50)
        logger.info("Initializing OncologyLiteratureMiner Pipeline")
        logger.info("="*50)
        self.collector = LiteratureCollector()
        self.processor = TextProcessor()
        self.extractor = RelationshipExtractor(self.processor)
        self.graph = KnowledgeGraph(self.processor)
        logger.info("Pipeline components initialized successfully.")
        
    def run_pubmed_pipeline(self, query: str = "cancer AND (immunotherapy OR targeted therapy)", 
                          max_results: int = 500) -> None:
        """
        Run complete pipeline on PubMed data
        Args:
            query: PubMed search query
            max_results: Maximum number of articles to process
        """
        logger.info("\n" + "="*50)
        logger.info("Starting PubMed Pipeline")
        logger.info("="*50)
        
        try:
            # Step 1: Collect articles
            articles = self.collector.search_pubmed(query, max_results)
            if not articles:
                logger.warning("No articles found or able to be processed.")
                return
                
            logger.info(f"\nProcessing {len(articles)} articles...")
            
            # Step 2: Process articles and extract knowledge
            for i, article in enumerate(tqdm(articles, desc="Processing articles")):
                try:
                    if i % 100 == 0:
                        logger.info(f"Processing article {i+1}/{len(articles)}: {article.get('title', '')[:50]}...")
                    
                    # Combine title and abstract
                    text = f"{article.get('title', '')}. {article.get('abstract', '')}"
                    
                    # Step 3: Extract entities
                    entities = self.processor.extract_entities(text)
                    
                    # Step 4: Extract relationships
                    relationships = self.extractor.extract_relationships(text, entities)
                    
                    # Step 5: Add to knowledge graph
                    for rel in relationships:
                        rel.update({
                            'source_id': article.get('pmid'),
                            'source_title': article.get('title'),
                            'source_journal': article.get('journal'),
                            'source_date': article.get('pub_date')
                        })
                        self.graph.add_relationship(rel)
                        
                except Exception as e:
                    logger.error(f"Error processing article {i+1}: {str(e)}")
                    continue
                    
            logger.info(f"\nKnowledge graph built with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")
            
        except Exception as e:
            logger.error(f"Fatal error in PubMed pipeline: {str(e)}", exc_info=True)

    def run_clinical_trials_pipeline(self, condition: str = "breast cancer", 
                                   intervention: Optional[str] = None,
                                   max_results: int = 100) -> None:
        """
        Run complete pipeline on clinical trials data
        Args:
            condition: Medical condition to search for
            intervention: Optional intervention filter
            max_results: Maximum number of trials to process
        """
        logger.info("\n" + "="*50)
        logger.info("Starting Clinical Trials Pipeline")
        logger.info("="*50)
        
        try:
            # Step 1: Collect trials
            trials = self.collector.search_clinical_trials(condition, intervention, max_results)
            if not trials:
                logger.warning("No trials found matching criteria.")
                return
                
            logger.info(f"\nProcessing {len(trials)} trials...")
            
            # Step 2: Process trials and extract knowledge
            for i, trial in enumerate(tqdm(trials, desc="Processing trials")):
                try:
                    if i % 20 == 0:
                        logger.info(f"Processing trial {i+1}/{len(trials)}: {trial.get('title', '')[:50]}...")
                    
                    # Combine relevant text fields
                    text = f"{trial.get('title', '')}. {trial.get('description', '')}"
                    if trial.get('interventions'):
                        text += ". Interventions: " + ", ".join(trial['interventions'])
                    
                    # Step 3: Extract entities
                    entities = self.processor.extract_entities(text)
                    
                    # Step 4: Extract relationships
                    relationships = self.extractor.extract_relationships(text, entities)
                    
                    # Step 5: Add to knowledge graph
                    for rel in relationships:
                        rel.update({
                            'source_id': trial.get('nct_id'),
                            'source_title': trial.get('title'),
                            'source_phase': trial.get('phase'),
                            'source_status': trial.get('status')
                        })
                        self.graph.add_relationship(rel)
                        
                except Exception as e:
                    logger.error(f"Error processing trial {i+1}: {str(e)}")
                    continue
                    
            logger.info(f"\nKnowledge graph now has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")
            
        except Exception as e:
            logger.error(f"Fatal error in clinical trials pipeline: {str(e)}", exc_info=True)
        
    def run_patents_pipeline(self, query: str = "cancer immunotherapy", 
                           max_results: int = 50) -> None:
        """
        Run pipeline on patent data
        Args:
            query: Patents search query
            max_results: Maximum number of patents to process
        """
        logger.info("\n" + "="*50)
        logger.info("Starting Patents Pipeline")
        logger.info("="*50)
        
        try:
            # Step 1: Collect patents
            patents = self.collector.search_google_patents(query, max_results)
            if not patents:
                logger.warning("No patents found matching criteria.")
                return
                
            logger.info(f"\nProcessing {len(patents)} patents...")
            
            # Step 2: Process patents and extract knowledge
            for i, patent in enumerate(tqdm(patents, desc="Processing patents")):
                try:
                    if i % 10 == 0:
                        logger.info(f"Processing patent {i+1}/{len(patents)}: {patent.get('title', '')[:50]}...")
                    
                    # Combine relevant text fields
                    text = f"{patent.get('title', '')}. {patent.get('abstract', '')}"
                    if patent.get('claims'):
                        text += ". Claims: " + ". ".join(patent['claims'])
                    
                    # Step 3: Extract entities
                    entities = self.processor.extract_entities(text)
                    
                    # Step 4: Extract relationships
                    relationships = self.extractor.extract_relationships(text, entities)
                    
                    # Add specialized patent relationships
                    relationships.extend(self.extractor.extract_drug_combinations(text))
                    
                    # Step 5: Add to knowledge graph
                    for rel in relationships:
                        rel.update({
                            'source_id': patent.get('patent_id'),
                            'source_title': patent.get('title'),
                            'source_date': patent.get('publication_date'),
                            'source_type': 'patent'
                        })
                        self.graph.add_relationship(rel)
                        
                except Exception as e:
                    logger.error(f"Error processing patent {i+1}: {str(e)}")
                    continue
                    
            logger.info(f"\nKnowledge graph now has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")
            
        except Exception as e:
            logger.error(f"Fatal error in patents pipeline: {str(e)}", exc_info=True)
            
    def analyze_network(self) -> None:
        """Analyze and display network metrics"""
        logger.info("\nAnalyzing knowledge graph...")
        try:
            metrics = self.graph.analyze_network()
            
            logger.info("\nNetwork Metrics:")
            logger.info(f"- Nodes: {metrics['num_nodes']}")
            logger.info(f"- Edges: {metrics['num_edges']}")
            logger.info(f"- Density: {metrics['density']:.4f}")
            
            if metrics.get('degree_centrality'):
                logger.info("\nTop Nodes by Degree Centrality:")
                for node, score in metrics['degree_centrality'].items():
                    logger.info(f"  {node}: {score:.3f}")
                    
            if metrics.get('betweenness_centrality'):
                logger.info("\nTop Nodes by Betweenness Centrality:")
                for node, score in metrics['betweenness_centrality'].items():
                    logger.info(f"  {node}: {score:.3f}")
                    
        except Exception as e:
            logger.error(f"Error analyzing network: {str(e)}")
    
    def save_results(self, filename_prefix: str = "oncology_knowledge_graph") -> None:
        """
        Save results to CSV and JSON files
        Args:
            filename_prefix: Prefix for output files
        """
        logger.info("\nSaving results to files...")
        
        # Create output directory if needed
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Save as CSV
            nodes_df, edges_df = self.graph.to_dataframe()
            
            nodes_csv = f"{output_dir}/{filename_prefix}_nodes.csv"
            edges_csv = f"{output_dir}/{filename_prefix}_edges.csv"
            
            nodes_df.to_csv(nodes_csv, index=False)
            edges_df.to_csv(edges_csv, index=False)
            
            # Save as JSON (full graph)
            graph_json = f"{output_dir}/{filename_prefix}_graph.json"
            graph_data = {
                'nodes': nodes_df.to_dict('records'),
                'edges': edges_df.to_dict('records')
            }
            with open(graph_json, 'w') as f:
                json.dump(graph_data, f, indent=2)
                
            logger.info(f"Results saved to:\n- {nodes_csv}\n- {edges_csv}\n- {graph_json}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
        
    def visualize_graph(self, max_nodes: int = 75) -> None:
        """
        Visualize the knowledge graph
        Args:
            max_nodes: Maximum number of nodes to display
        """
        logger.info("\nGenerating knowledge graph visualization...")
        try:
            self.graph.visualize(max_nodes=max_nodes)
        except Exception as e:
            logger.error(f"Error visualizing graph: {str(e)}")

# Example Usage
if __name__ == "__main__":
    logger.info("\n" + "="*50)
    logger.info("Starting Oncology Literature Mining MVP")
    logger.info("="*50)

    try:
        # Initialize pipeline
        miner = OncologyLiteratureMiner()
        
        # Run PubMed pipeline with custom query
        query = miner.collector.build_pubmed_query(
            drugs=['pembrolizumab', 'nivolumab'],
            biomarkers=['PD-L1'],
            cancers=['non-small cell lung cancer'],
            date_range=('2020/01/01', '2023/12/31')
        )
        miner.run_pubmed_pipeline(query=query, max_results=500)
        
        # Run clinical trials pipeline
        miner.run_clinical_trials_pipeline(condition="non-small cell lung cancer")
        
        # Run patents pipeline
        miner.run_patents_pipeline(query="cancer immunotherapy PD-L1")
        
        # Analyze and save results
        miner.analyze_network()
        miner.save_results()
        miner.graph.save_gexf()
        
        # Visualize
        miner.visualize_graph(max_nodes=100)
        
        logger.info("\n" + "="*50)
        logger.info("Pipeline completed successfully!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Fatal error in pipeline execution: {str(e)}", exc_info=True)


