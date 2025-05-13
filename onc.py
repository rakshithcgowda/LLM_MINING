import requests
import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
from tqdm import tqdm
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
    logger.warning("Scientific model not found, falling back to other models...")
    for model in ["en_core_web_md", "en_core_web_sm"]:
        try:
            nlp = spacy.load(model)
            logger.info(f"Loaded {model} model successfully.")
            break
        except OSError:
            continue

# Constants
CACHE_DIR = "./data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
logger.info(f"Using cache directory: {CACHE_DIR}")

# Configure requests session
session = requests.Session()
session.headers.update({ 'User-Agent': 'OncologyLiteratureMiner/1.0' })

class LiteratureCollector:
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
        cache_file = os.path.join(CACHE_DIR, f"pubmed_{hash(query)}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    if data:
                        logger.info(f"Loaded {len(data)} articles from cache")
                        return data
            except:
                pass

        params = {
            'db': 'pubmed', 'term': query,
            'retmax': min(max_results, 10000), 'retmode': 'json',
            'sort': 'relevance', 'field': 'title/abstract',
            'mindate': '2010/01/01', 'api_key': os.getenv('NCBI_API_KEY','')
        }
        logger.info(f"Searching PubMed: {query}")
        resp = self._make_request(self.base_urls['pubmed'] + 'esearch.fcgi', params)
        if not resp: return []
        pmids = resp.json().get('esearchresult', {}).get('idlist', [])
        if not pmids:
            logger.info("No PubMed results.")
            return []

        articles = []
        for i in range(0, len(pmids), 100):
            batch = pmids[i:i+100]
            params = {'db':'pubmed','id':','.join(batch),'retmode':'xml','api_key':os.getenv('NCBI_API_KEY','')}
            resp = self._make_request(self.base_urls['pubmed'] + 'efetch.fcgi', params, is_xml=True)
            if not resp: continue
            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError:
                content = resp.content
                if content.startswith(b'<?xml'): content = content.split(b'?>',1)[1]
                root = ET.fromstring(content)
            for art in root.findall('.//PubmedArticle'):
                try:
                    pmid = art.findtext('.//PMID')
                    title = art.findtext('.//ArticleTitle')
                    abstract = ' '.join([t.text for t in art.findall('.//AbstractText') if t.text])
                    journal = art.findtext('.//Journal/Title')
                    date = '-'.join(filter(None,[art.findtext('.//PubDate/Year'), art.findtext('.//PubDate/Month'), art.findtext('.//PubDate/Day')]))
                    doi = art.findtext('.//ArticleId[@IdType="doi"]')
                    pmcid = art.findtext('.//ArticleId[@IdType="pmc"]')
                    authors = [f"{a.findtext('LastName')} {a.findtext('ForeName')}" for a in art.findall('.//Author') if a.findtext('LastName')]
                    articles.append({'pmid':pmid,'title':title,'abstract':abstract,'journal':journal,'pub_date':date,'doi':doi,'pmcid':pmcid,'authors':authors})
                except Exception:
                    continue
            time.sleep(0.5)

        if articles:
            with open(cache_file,'w') as f: json.dump(articles,f,indent=2)
        return articles

    def search_clinical_trials(self, condition: str, intervention: str = None, max_results: int = 100) -> List[Dict]:
        cache_file = os.path.join(CACHE_DIR, f"clinical_{hash(condition+str(intervention))}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    if data: return data
            except:
                pass

        params = {'query.cond': condition, 'format':'json','pageSize':min(max_results,1000)}
        if intervention: params['query.intr'] = intervention
        logger.info(f"Searching ClinicalTrials.gov: {condition}, intervention={intervention}")
        resp = self._make_request(self.base_urls['clinical_trials'], params)
        if not resp: return []
        studies = resp.json().get('studies', [])
        trials = []
        for st in studies:
            sec = st.get('protocolSection',{})
            idm = sec.get('identificationModule',{})
            des = sec.get('descriptionModule',{})
            stat = sec.get('statusModule',{})
            desi = sec.get('designModule',{})
            arms = sec.get('armsInterventionsModule',{})
            trials.append({
                'nct_id':idm.get('nctId'), 'title':idm.get('briefTitle'),
                'official_title':idm.get('officialTitle'), 'description':des.get('briefSummary'),
                'conditions':[c.get('name') for c in desi.get('conditions',[])],
                'interventions':[i.get('name') for i in arms.get('interventions',[])],
                'phase':desi.get('phase'), 'status':stat.get('overallStatus'),
                'start_date':stat.get('startDate'), 'completion_date':stat.get('completionDate'),
                'study_type':desi.get('studyType'), 'enrollment':desi.get('enrollmentInfo',{}).get('count')
            })
        if trials:
            with open(cache_file,'w') as f: json.dump(trials,f,indent=2)
        return trials
    processed_trials

# Rest of classes (TextProcessor, RelationshipExtractor, KnowledgeGraph, OncologyLiteratureMiner) unchanged...
# In the main execution, change the query to remove duplicate "biomarker":

if __name__ == "__main__":
    miner = OncologyLiteratureMiner()
    query = "cancer AND (immunotherapy OR targeted therapy) AND biomarker"
    miner.run_pubmed_pipeline(query=query, max_results=500)
    if input("Run clinical trials? (y/n): ").lower()=='y':
        cond = input("Condition: ") or "breast cancer"
        intr = input("Intervention (optional): ") or None
        miner.run_clinical_trials_pipeline(condition=cond, intervention=intr)
    miner.save_results()
    if input("Visualize graph? (y/n): ").lower()=='y':
        miner.visualize_graph()
