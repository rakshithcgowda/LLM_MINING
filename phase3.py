#!/usr/bin/env python3
"""
Oncology Knowledge Graph System - Production Ready Implementation

A comprehensive system for mining, analyzing, and querying oncology literature
and clinical trial data to build a knowledge graph of cancer treatments,
biomarkers, mutations, and their relationships.
"""

import os
import re
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Set, Any, Union
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import requests
import spacy
import networkx as nx
from tqdm import tqdm
from pydantic import BaseModel, Field, validator
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing_extensions import Literal
import xml.etree.ElementTree as ET

# Configuration
class Config(BaseModel):
    """Application configuration model"""
    cache_dir: str = "./data_cache"
    log_level: str = "INFO"
    max_retries: int = 3
    retry_delay: float = 2.0
    api_timeout: int = 30
    max_results: int = 1000
    graph_visualization_max_nodes: int = 100
    nlp_model: str = "en_core_sci_sm"
    fallback_nlp_models: List[str] = ["en_core_web_md", "en_core_web_sm"]
    enable_apis: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        env_prefix = "ONCOKG_"

# Initialize configuration
config = Config()

# Set up logging
logging.basicConfig(
    level=config.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("oncology_kg.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure cache directory exists
os.makedirs(config.cache_dir, exist_ok=True)

# Initialize NLP model
def load_nlp_model():
    """Load the best available NLP model"""
    models_to_try = [config.nlp_model] + config.fallback_nlp_models
    for model_name in models_to_try:
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Successfully loaded NLP model: {model_name}")
            return nlp
        except OSError:
            logger.warning(f"Model {model_name} not found, trying next...")
            continue
    
    raise ImportError("Could not load any of the specified NLP models")

nlp = load_nlp_model()

# Data Models
class Entity(BaseModel):
    """Base entity model"""
    id: str
    label: str
    type: str
    attributes: Dict[str, Any] = Field(default_factory=dict)

class Relationship(BaseModel):
    """Relationship between entities"""
    source_id: str
    target_id: str
    type: str
    evidence: str
    score: float = Field(..., ge=0, le=1)
    attributes: Dict[str, Any] = Field(default_factory=dict)

class KnowledgeGraph(BaseModel):
    """Knowledge graph container"""
    nodes: Dict[str, Entity] = Field(default_factory=dict)
    edges: List[Relationship] = Field(default_factory=list)

class QueryRequest(BaseModel):
    """Query request model"""
    query_type: Literal["drugs_by_mutation", "biomarkers_by_drug", "clinical_trials"]
    mutation: Optional[str] = None
    drug: Optional[str] = None
    biomarker: Optional[str] = None
    cancer_type: Optional[str] = None
    phase: Optional[str] = None
    limit: int = 10

    @validator('query_type')
    def validate_query_type(cls, v, values):
        if v == "drugs_by_mutation" and not values.get('mutation'):
            raise ValueError("mutation is required for drugs_by_mutation query")
        if v == "biomarkers_by_drug" and not values.get('drug'):
            raise ValueError("drug is required for biomarkers_by_drug query")
        return v

# Core Components
class LiteratureCollector:
    """Handles collection of literature from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'OncologyLiteratureMiner/1.0',
            'Accept': 'application/json'
        })
        self.base_urls = {
            'pubmed': "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            'clinical_trials': "https://clinicaltrials.gov/api/v2/studies",
            'google_patents': "https://patents.googleapis.com/v1/patents:search"
        }

    def _make_request(self, url: str, params: Dict, is_xml: bool = False) -> Optional[requests.Response]:
        """Make HTTP request with retries"""
        for attempt in range(config.max_retries):
            try:
                resp = self.session.get(
                    url,
                    params=params,
                    timeout=config.api_timeout
                )
                resp.raise_for_status()
                if is_xml and not resp.content.strip():
                    raise ValueError("Empty XML response")
                return resp
            except Exception as e:
                if attempt == config.max_retries - 1:
                    logger.error(f"Request failed after {config.max_retries} attempts: {e}")
                    return None
                time.sleep(config.retry_delay)

    def build_pubmed_query(self, drugs: List[str] = None, 
                         biomarkers: List[str] = None,
                         cancers: List[str] = None,
                         genes: List[str] = None,
                         date_range: Tuple[str, str] = None) -> str:
        """Build a sophisticated PubMed query"""
        query_parts = []
        
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
            
        if date_range:
            start, end = date_range
            query_parts.append(f'("{start}"[Date - Publication] : "{end}"[Date - Publication])')
            
        return ' AND '.join(query_parts) if query_parts else 'cancer'

    def search_pubmed(self, query: str, max_results: int = config.max_results) -> List[Dict]:
        """Search PubMed and return article metadata"""
        cache_key = f"pubmed_{hash(query)}"
        cache_file = os.path.join(config.cache_dir, f"{cache_key}.json")
        
        # Check cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Execute search
        search_params = {
            'db': 'pubmed',
            'term': query,
            'retmax': min(max_results, 10000),
            'retmode': 'json',
        }
        
        if os.getenv('NCBI_API_KEY'):
            search_params['api_key'] = os.getenv('NCBI_API_KEY')

        response = self._make_request(
            self.base_urls['pubmed'] + 'esearch.fcgi',
            search_params
        )
        if not response:
            return []

        data = response.json()
        pmids = data.get('esearchresult', {}).get('idlist', [])
        if not pmids:
            return []

        # Fetch details in batches
        articles = []
        for i in range(0, len(pmids), 100):
            batch = pmids[i:i+100]
            fetch_params = {
                'db': 'pubmed',
                'id': ",".join(batch),
                'retmode': 'xml'
            }
            fetch_resp = self._make_request(
                self.base_urls['pubmed'] + 'efetch.fcgi',
                fetch_params,
                is_xml=True
            )
            if not fetch_resp:
                continue

            try:
                root = ET.fromstring(fetch_resp.content)
            except ET.ParseError:
                content = fetch_resp.content
                if content.startswith(b'<?xml'):
                    content = content.split(b'?>', 1)[1]
                root = ET.fromstring(content)

            for art in root.findall('.//PubmedArticle'):
                try:
                    articles.append({
                        'pmid': art.findtext('.//PMID'),
                        'title': art.findtext('.//ArticleTitle'),
                        'abstract': ' '.join([t.text for t in art.findall('.//AbstractText') if t.text]),
                        'journal': art.findtext('.//Journal/Title'),
                        'pub_date': self._parse_pub_date(art),
                        'doi': art.findtext('.//ArticleId[@IdType="doi"]'),
                        'pmcid': art.findtext('.//ArticleId[@IdType="pmc"]'),
                        'authors': self._parse_authors(art)
                    })
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue

            time.sleep(0.5)

        # Cache results
        if articles:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(articles, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to cache results: {e}")

        return articles

    def _parse_pub_date(self, article) -> str:
        """Parse publication date from article XML"""
        pub_date = article.find('.//PubDate')
        if pub_date is None:
            return ""
        
        year = pub_date.findtext('Year', '').strip()
        month = pub_date.findtext('Month', '').strip()
        day = pub_date.findtext('Day', '').strip()
        
        if year:
            return f"{year}-{month}-{day}" if month and day else year
        return ""

    def _parse_authors(self, article) -> List[str]:
        """Parse authors from article XML"""
        authors = []
        for author in article.findall('.//Author'):
            last_name = author.findtext('LastName', '').strip()
            fore_name = author.findtext('ForeName', '').strip()
            if last_name and fore_name:
                authors.append(f"{last_name} {fore_name}")
        return authors

class TextProcessor:
    """Handles text processing and entity extraction"""
    
    def __init__(self):
        self.cancer_types = self._load_cancer_types()
        self.drug_lexicon = self._load_drug_lexicon()
        self.biomarkers = self._load_biomarkers()
        self.genes = self._load_genes()
        self.mutations = self._load_mutations()
        
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
        
    def _load_drug_lexicon(self) -> Dict[str, Dict]:
        """Load drug names and categories with mechanisms"""
        return {
            'trastuzumab': {
                'type': 'targeted therapy',
                'mechanism': 'HER2 receptor antagonist',
                'target': 'ERBB2 (HER2)'
            },
            'pembrolizumab': {
                'type': 'immunotherapy',
                'mechanism': 'PD-1 inhibitor',
                'target': 'PDCD1 (PD-1)'
            },
            # Add more drugs with mechanisms
        }
        
    def _load_biomarkers(self) -> Set[str]:
        """Load known biomarkers"""
        return {
            'PD-L1', 'HER2', 'EGFR', 'ALK', 'BRAF', 'MSI', 'TMB', 'TP53',
            'BRCA1', 'BRCA2', 'KRAS', 'NRAS', 'PIK3CA', 'PTEN', 'ROS1'
        }
        
    def _load_genes(self) -> Set[str]:
        """Load cancer-related genes"""
        return {
            'EGFR', 'HER2', 'KRAS', 'NRAS', 'BRAF', 'ALK', 'ROS1', 'MET',
            'BRCA1', 'BRCA2', 'TP53', 'PTEN', 'PIK3CA', 'MYC', 'CDKN2A'
        }
        
    def _load_mutations(self) -> Dict[str, Dict]:
        """Load common cancer mutations with types"""
        return {
            'EGFR L858R': {
                'type': 'missense',
                'domain': 'kinase',
                'effect': 'activation'
            },
            'TP53 R175H': {
                'type': 'missense',
                'domain': 'DNA-binding',
                'effect': 'loss of function'
            }
            # Add more mutations
        }
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract biomedical entities from text"""
        if not text:
            return {}
            
        text = self._preprocess_text(text)
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
                
        # Deduplicate
        for key in entities:
            entities[key] = sorted(list(set(entities[key])))
            
        return entities
        
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\[[\d,]+\]', '', text)  # Remove citations
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)  # Keep basic punctuation
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

class RelationshipExtractor:
    """Extracts relationships between entities"""
    
    def __init__(self, processor: TextProcessor):
        self.processor = processor
        self.patterns = self._init_relationship_patterns()
        
    def _init_relationship_patterns(self) -> Dict[str, List[Tuple[str, re.Pattern]]]:
        """Define regex patterns for relationship extraction"""
        patterns = {
            'drug_target': [
                (r'(?P<drug>\w+)\s+(?:targets?|inhibits?|blocks?)\s+(?P<target>\w+)', re.IGNORECASE),
                (r'(?P<target>\w+)\s+(?:is targeted by|is inhibited by)\s+(?P<drug>\w+)', re.IGNORECASE)
            ],
            'biomarker_response': [
                (r'(?P<biomarker>\w+)[-\s]?(?:positive|status)\s+(?:predicts|correlates with)\s+response to (?P<drug>\w+)', re.IGNORECASE)
            ],
            'drug_combination': [
                (r'(?P<drug1>\w+)\s*\+\s*(?P<drug2>\w+)\s+(?:regimen|therapy)', re.IGNORECASE)
            ]
        }
        
        compiled = {}
        for rel_type, pattern_list in patterns.items():
            compiled[rel_type] = []
            for pattern_str, flags in pattern_list:
                try:
                    compiled[rel_type].append(
                        (pattern_str, re.compile(pattern_str, flags))
                    )
                except re.error as e:
                    logger.error(f"Error compiling pattern: {e}")
                    
        return compiled
        
    def extract_relationships(self, text: str, entities: Dict[str, List[str]]) -> List[Dict]:
        """Extract relationships from text"""
        relationships = []
        text_lower = text.lower()
        
        # Co-occurrence relationships
        self._add_cooccurrence_rels(relationships, entities, text)
        
        # Pattern-based relationships
        for rel_type, pattern_list in self.patterns.items():
            for pattern_str, pattern in pattern_list:
                for match in pattern.finditer(text_lower):
                    groups = match.groupdict()
                    if len(groups) == 2:
                        relationships.append({
                            'type': rel_type,
                            'entity1': groups[list(groups.keys())[0]],
                            'entity2': groups[list(groups.keys())[1]],
                            'evidence': text[match.start():match.end()+100],
                            'source': 'pattern',
                            'score': 0.8
                        })
        
        # Validate against known entities
        return [r for r in relationships if self._validate_relationship(r, entities)]
        
    def _add_cooccurrence_rels(self, relationships: List[Dict], entities: Dict[str, List[str]], text: str) -> None:
        """Add co-occurrence based relationships"""
        # Drug-drug
        if len(entities['drugs']) >= 2:
            for i in range(len(entities['drugs'])):
                for j in range(i+1, len(entities['drugs'])):
                    relationships.append({
                        'type': 'drug_combination',
                        'entity1': entities['drugs'][i],
                        'entity2': entities['drugs'][j],
                        'evidence': f"Co-occurrence: {text[:200]}...",
                        'source': 'cooccurrence',
                        'score': 0.6
                    })
                    
        # Drug-gene
        if entities['drugs'] and entities['genes']:
            for drug in entities['drugs']:
                for gene in entities['genes']:
                    relationships.append({
                        'type': 'drug_target',
                        'entity1': drug,
                        'entity2': gene,
                        'evidence': f"Co-occurrence: {text[:200]}...",
                        'source': 'cooccurrence',
                        'score': 0.7
                    })
                    
    def _validate_relationship(self, rel: Dict, entities: Dict[str, List[str]]) -> bool:
        """Validate that relationship entities exist in extracted entities"""
        ent1 = rel['entity1'].lower()
        ent2 = rel['entity2'].lower()
        
        ent1_valid = any(ent1 in [e.lower() for e in entities[cat]] 
                        for cat in ['drugs', 'genes', 'biomarkers', 'cancer_types', 'mutations'])
        ent2_valid = any(ent2 in [e.lower() for e in entities[cat]] 
                        for cat in ['drugs', 'genes', 'biomarkers', 'cancer_types', 'mutations'])
        
        return ent1_valid and ent2_valid

class OncologyKnowledgeGraph:
    """Main knowledge graph class with enhanced attributes"""
    
    def __init__(self):
        self.collector = LiteratureCollector()
        self.processor = TextProcessor()
        self.extractor = RelationshipExtractor(self.processor)
        self.graph = KnowledgeGraph()
        self.node_counter = 0
        self.edge_counter = 0
        
    def build_from_pubmed(self, query: str, max_results: int = config.max_results) -> None:
        """Build graph from PubMed articles"""
        articles = self.collector.search_pubmed(query, max_results)
        
        for article in tqdm(articles, desc="Processing articles"):
            text = f"{article.get('title', '')}. {article.get('abstract', '')}"
            
            # Extract entities and relationships
            entities = self.processor.extract_entities(text)
            relationships = self.extractor.extract_relationships(text, entities)
            
            # Add to graph
            for rel in relationships:
                self._add_relationship(rel, article)
                
    def _add_relationship(self, relationship: Dict, source: Dict = None) -> None:
        """Add a relationship to the graph"""
        # Add nodes
        for entity in [relationship['entity1'], relationship['entity2']]:
            if entity not in self.graph.nodes:
                self._add_node(entity)
                
        # Create edge
        edge_id = f"e{self.edge_counter}"
        self.edge_counter += 1
        
        edge = Relationship(
            source_id=relationship['entity1'],
            target_id=relationship['entity2'],
            type=relationship['type'],
            evidence=relationship.get('evidence', ''),
            score=relationship.get('score', 0.5),
            attributes={
                'source': relationship.get('source', 'unknown'),
                'source_id': source.get('pmid') if source else None,
                'source_title': source.get('title') if source else None
            }
        )
        
        self.graph.edges.append(edge)
        
    def _add_node(self, label: str) -> None:
        """Add a node to the graph with attributes"""
        node_id = f"n{self.node_counter}"
        self.node_counter += 1
        
        # Determine node type and attributes
        node_type, attributes = self._get_node_type_and_attributes(label)
        
        node = Entity(
            id=node_id,
            label=label,
            type=node_type,
            attributes=attributes
        )
        
        self.graph.nodes[label] = node
        
    def _get_node_type_and_attributes(self, label: str) -> Tuple[str, Dict]:
        """Determine node type and attributes based on label"""
        label_lower = label.lower()
        
        # Check drug lexicon first
        if label_lower in self.processor.drug_lexicon:
            return 'drug', self.processor.drug_lexicon[label_lower]
            
        # Check other entity types
        if label in self.processor.biomarkers:
            return 'biomarker', {'category': 'biomarker'}
            
        if label in self.processor.genes:
            return 'gene', {'category': 'gene'}
            
        if any(label_lower == ct.lower() for ct in self.processor.cancer_types):
            return 'cancer_type', {'category': 'disease'}
            
        if any(mut.lower() in label_lower for mut in self.processor.mutations):
            mutation_data = next(
                (self.processor.mutations[m] for m in self.processor.mutations 
                if m.lower() in label_lower),
                {}
            )
            return 'mutation', mutation_data
            
        return 'other', {'category': 'other'}

class QueryEngine:
    """Query engine for the knowledge graph"""
    
    def __init__(self, graph: OncologyKnowledgeGraph):
        self.graph = graph
        
    def find_drugs_by_mutation(self, mutation: str, cancer_type: str = None) -> List[Dict]:
        """Find drugs targeting a specific mutation"""
        results = []
        
        for edge in self.graph.graph.edges:
            if edge.type in ['drug_target', 'drug_combination']:
                if (mutation.lower() in edge.source_id.lower() or 
                    mutation.lower() in edge.target_id.lower()):
                    
                    # Determine which node is the drug
                    drug_node = None
                    other_node = None
                    
                    if edge.source_id.lower() in self.graph.processor.drug_lexicon:
                        drug_node = edge.source_id
                        other_node = edge.target_id
                    elif edge.target_id.lower() in self.graph.processor.drug_lexicon:
                        drug_node = edge.target_id
                        other_node = edge.source_id
                        
                    if drug_node:
                        # Filter by cancer type if specified
                        if cancer_type:
                            cancer_found = False
                            for e in self.graph.graph.edges:
                                if (e.source_id.lower() == cancer_type.lower() and 
                                    e.target_id == drug_node) or \
                                   (e.target_id.lower() == cancer_type.lower() and 
                                    e.source_id == drug_node):
                                    cancer_found = True
                                    break
                            if not cancer_found:
                                continue
                                
                        results.append({
                            'drug': drug_node,
                            'target': other_node,
                            'relationship': edge.type,
                            'evidence': edge.evidence,
                            'score': edge.score,
                            'source': edge.attributes.get('source_title', '')
                        })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
        
    def find_biomarkers_by_drug(self, drug: str) -> List[Dict]:
        """Find biomarkers associated with a drug"""
        results = []
        drug_lower = drug.lower()
        
        for edge in self.graph.graph.edges:
            if edge.type == 'biomarker_response':
                if (edge.source_id.lower() == drug_lower or 
                    edge.target_id.lower() == drug_lower):
                    
                    # Determine which node is the biomarker
                    biomarker_node = None
                    if edge.source_id in self.graph.processor.biomarkers:
                        biomarker_node = edge.source_id
                    elif edge.target_id in self.graph.processor.biomarkers:
                        biomarker_node = edge.target_id
                        
                    if biomarker_node:
                        results.append({
                            'drug': drug,
                            'biomarker': biomarker_node,
                            'evidence': edge.evidence,
                            'score': edge.score,
                            'source': edge.attributes.get('source_title', '')
                        })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

# API Implementation
app = FastAPI(
    title="Oncology Knowledge Graph API",
    description="API for querying oncology treatment relationships",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global graph instance
knowledge_graph = None
query_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the knowledge graph on startup"""
    global knowledge_graph, query_engine
    
    logger.info("Initializing Oncology Knowledge Graph...")
    knowledge_graph = OncologyKnowledgeGraph()
    
    # Build with example query
    query = knowledge_graph.collector.build_pubmed_query(
        drugs=['pembrolizumab', 'nivolumab', 'trastuzumab'],
        biomarkers=['PD-L1', 'HER2'],
        cancers=['non-small cell lung cancer', 'breast cancer']
    )
    knowledge_graph.build_from_pubmed(query, max_results=500)
    
    query_engine = QueryEngine(knowledge_graph)
    logger.info(f"Knowledge graph initialized with {len(knowledge_graph.graph.nodes)} nodes and {len(knowledge_graph.graph.edges)} edges")

@app.post("/query", response_model=List[Dict])
async def query_knowledge_graph(request: QueryRequest):
    """Query the knowledge graph"""
    try:
        if request.query_type == "drugs_by_mutation":
            results = query_engine.find_drugs_by_mutation(
                request.mutation,
                request.cancer_type
            )
        elif request.query_type == "biomarkers_by_drug":
            results = query_engine.find_biomarkers_by_drug(request.drug)
        else:
            raise HTTPException(status_code=400, detail="Invalid query type")
            
        return results[:request.limit]
        
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph_stats", response_model=Dict)
async def get_graph_stats():
    """Get basic graph statistics"""
    if not knowledge_graph:
        raise HTTPException(status_code=503, detail="Graph not initialized")
        
    return {
        "node_count": len(knowledge_graph.graph.nodes),
        "edge_count": len(knowledge_graph.graph.edges),
        "node_types": defaultdict(int, [
            (node.type, 1) for node in knowledge_graph.graph.nodes.values()
        ]),
        "edge_types": defaultdict(int, [
            (edge.type, 1) for edge in knowledge_graph.graph.edges
        ])
    }

def run_api():
    """Run the FastAPI server"""
    uvicorn.run(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower()
    )

# Command Line Interface
def main():
    """Main entry point for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Oncology Knowledge Graph System"
    )
    parser.add_argument(
        '--build',
        action='store_true',
        help='Build the knowledge graph from PubMed'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Run a query (e.g., "drugs_for_mutation:EGFR")'
    )
    parser.add_argument(
        '--api',
        action='store_true',
        help='Start the API server'
    )
    
    args = parser.parse_args()
    
    if args.build:
        kg = OncologyKnowledgeGraph()
        query = kg.collector.build_pubmed_query(
            drugs=['osimertinib', 'gefitinib'],
            biomarkers=['EGFR'],
            cancers=['non-small cell lung cancer']
        )
        kg.build_from_pubmed(query)
        print(f"Built graph with {len(kg.graph.nodes)} nodes and {len(kg.graph.edges)} edges")
        
    elif args.query:
        kg = OncologyKnowledgeGraph()
        query_engine = QueryEngine(kg)
        
        if args.query.startswith("drugs_for_mutation:"):
            mutation = args.query.split(":")[1]
            results = query_engine.find_drugs_by_mutation(mutation)
            for r in results[:10]:
                print(f"{r['drug']} (target: {r['target']}, score: {r['score']:.2f})")
                
        elif args.query.startswith("biomarkers_for_drug:"):
            drug = args.query.split(":")[1]
            results = query_engine.find_biomarkers_by_drug(drug)
            for r in results[:10]:
                print(f"Biomarker: {r['biomarker']} (score: {r['score']:.2f})")
                
    elif args.api:
        run_api()
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()