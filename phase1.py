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
session.headers.update({ 'User-Agent': 'OncologyLiteratureMiner/1.0' })

class LiteratureCollector:
    """Handles collection of literature from various sources"""
    
    def __init__(self):
        self.base_urls = {
            'pubmed': "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            'clinical_trials': "https://clinicaltrials.gov/api/v2/studies"
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
        api_key = os.getenv('c0dd013bb98b52950819c622c3ce65adb908')
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
                'api_key': os.getenv('NCBI_API_KEY', 'c0dd013bb98b52950819c622c3ce65adb908')
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


    # Helper methods for PubMed XML processing
    def _safe_extract(self, element, xpath: str) -> Optional[str]:
        """Safely extract text from XML element"""
        found = element.find(xpath)
        return found.text if found is not None else None

    def _join_abstract_text(self, article) -> str:
        """Combine all abstract text sections"""
        abstract = article.find('.//Abstract')
        if abstract is None:
            return ""
        return ' '.join([
            elem.text for elem in abstract.findall('.//AbstractText') 
            if elem.text is not None
        ]).strip()

    def _extract_pub_date(self, article) -> str:
        """Extract publication date with fallbacks"""
        pub_date = article.find('.//PubDate')
        if pub_date is None:
            return ""
        
        date_parts = []
        for field in ['Year', 'Month', 'Day']:
            elem = pub_date.find(field)
            if elem is not None and elem.text:
                date_parts.append(elem.text)
        
        return '-'.join(date_parts) if date_parts else ""

    def _extract_authors(self, article) -> List[str]:
        """Extract author names in 'LastName FirstName' format"""
        authors = []
        for author in article.findall('.//AuthorList/Author'):
            last_name = author.find('LastName')
            fore_name = author.find('ForeName')
            if last_name is not None and last_name.text and fore_name is not None and fore_name.text:
                authors.append(f"{last_name.text} {fore_name.text}")
        return authors

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
        
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        if not text:
            return ""
            
        # Remove special characters, extra whitespace, and normalize
        text = re.sub(r'[^\w\s-]', ' ', text.lower())  # Keep words, spaces, and hyphens
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
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
    """Builds and manages the oncology knowledge graph"""
    
    def __init__(self, processor: TextProcessor):
        self.nodes = set()
        self.edges = []
        self.node_attributes = {}
        self.edge_attributes = {}
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
        for entity in [entity1, entity2]:
            if entity not in self.nodes:
                self.nodes.add(entity)
                self._set_node_attributes(entity)
                
        # Create edge key (undirected for co-occurrence, directed for others)
        if 'cooccurrence' in rel_type:
            edge_key = frozenset({entity1, entity2, rel_type})
        else:
            edge_key = (entity1, entity2, rel_type)
            
        # Check if edge already exists
        if any(self._edge_matches(edge_key, e) for e in self.edges):
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
        logger.debug(f"Added relationship: {entity1} -- {rel_type} -> {entity2}")
        
    def _edge_matches(self, edge_key, edge_data) -> bool:
        """Check if an edge matches the given key"""
        if isinstance(edge_key, frozenset):
            # For undirected edges (co-occurrence)
            return edge_key == frozenset({
                edge_data['source'], 
                edge_data['target'], 
                edge_data['type']
            })
        else:
            # For directed edges
            return (
                edge_key[0] == edge_data['source'] and
                edge_key[1] == edge_data['target'] and
                edge_key[2] == edge_data['type']
            )
            
    def _set_node_attributes(self, node: str) -> None:
        """Set attributes for a node based on its type"""
        node_lower = node.lower()
        attrs = {'id': f"n{self.node_counter}", 'label': node}
        self.node_counter += 1
        
        if node_lower in self.processor.drug_lexicon:
            attrs.update({
                'type': 'drug',
                'category': self.processor.drug_lexicon.get(node_lower, 'unknown'),
                'color': '#1f77b4'  # Blue
            })
        elif node in self.processor.biomarkers:
            attrs.update({
                'type': 'biomarker',
                'color': '#2ca02c'  # Green
            })
        elif node in self.processor.genes:
            attrs.update({
                'type': 'gene',
                'color': '#ff7f0e'  # Orange
            })
        elif node_lower in self.processor.cancer_types:
            attrs.update({
                'type': 'cancer_type',
                'color': '#d62728'  # Red
            })
        elif any(mut.lower() in node_lower for mut in self.processor.mutations):
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
        Convert knowledge graph to pandas DataFrames
        Returns:
            Tuple of (nodes_df, edges_df)
        """
        # Nodes DataFrame
        nodes_data = []
        for node in self.nodes:
            attrs = self.node_attributes.get(node, {})
            nodes_data.append({
                'id': attrs.get('id'),
                'label': node,
                'type': attrs.get('type', 'unknown'),
                'category': attrs.get('category', ''),
                'color': attrs.get('color', '#999999')
            })
        nodes_df = pd.DataFrame(nodes_data)
        
        # Edges DataFrame
        edges_df = pd.DataFrame(self.edges)
        
        return nodes_df, edges_df
        
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
            bbox_to_anchor=(1.1, 1.1)
        )
        
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
            intervention: Optional intervention to filter by
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
        
        # Run PubMed pipeline
        search_query = "cancer immunotherapy targeted therapy biomarker"
        miner.run_pubmed_pipeline(query=search_query, max_results=500)
        
        # Optionally run clinical trials pipeline
        run_clinical = input("\nWould you like to also run the clinical trials pipeline? (y/n): ").lower() == 'y'
        if run_clinical:
            condition = input("Enter cancer condition (e.g. 'breast cancer'): ").strip() or "breast cancer"
            intervention = input("Enter intervention (optional, press enter to skip): ").strip() or None
            miner.run_clinical_trials_pipeline(condition=condition, intervention=intervention)
        
        # Save results
        miner.save_results()
        
        # Visualize
        visualize = input("\nWould you like to visualize the knowledge graph? (y/n): ").lower() == 'y'
        if visualize:
            miner.visualize_graph()
        
        logger.info("\n" + "="*50)
        logger.info("Pipeline completed successfully!")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Fatal error in pipeline execution: {str(e)}", exc_info=True)