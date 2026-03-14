import requests
import torch
import heapq
import time
import logging
import json
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer
import numpy as np 
from vectortrain import AdvancedQNetwork 

# --- CONFIGURATION ---
USE_RL = False   # <--- TOGGLE THIS: True = Neural Brain, False = Greedy Baseline
START_SEED = "https://cs.nyu.edu/home/index.html"
TARGET_TOPIC = "Syllabus for Deep Learning Course"
CHECKPOINT_PATH = "checkpoint_advanced.pth"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_PAGES = 500
DELAY = 0.5 

OUTPUT_LOG = f"crawl_results_{'RL' if USE_RL else 'NAIVE'}.jsonl"
FRONTIER_LOG = f"frontier_snapshot_{'RL' if USE_RL else 'NAIVE'}.json"

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

class HybridCrawler:
    def __init__(self, use_rl=True):
        self.use_rl = use_rl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mode_name = "NEURAL RL" if self.use_rl else "NAIVE GREEDY"
        logger.info(f"Initializing Crawler in [{mode_name}] mode on {self.device}...")

        # 1. Load the Encoder (Needed for BOTH modes)
        self.encoder = SentenceTransformer(MODEL_NAME).to(self.device)
        
        # 2. Pre-compute Target Vector (The Compass)
        self.target_vec = self.encoder.encode(TARGET_TOPIC, convert_to_tensor=True).to(self.device)

        # 3. Load the Brain (ONLY for RL mode)
        if self.use_rl:
            self.brain = AdvancedQNetwork().to(self.device)
            try:
                # Robust checkpoint loading
                if os.path.exists(CHECKPOINT_PATH):
                    checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
                    if isinstance(checkpoint, dict) and 'q_network' in checkpoint:
                        self.brain.load_state_dict(checkpoint['q_network'])
                    else:
                        self.brain.load_state_dict(checkpoint)
                    self.brain.eval()
                    logger.info("Trained Q-Network Loaded.")
                else:
                    logger.warning(f"Checkpoint {CHECKPOINT_PATH} not found! Running with untrained weights.")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                raise e
        else:
            logger.info("Running in Baseline Mode (Cosine Similarity only).")

        self.frontier = []
        self.visited = set()
        self.session = requests.Session()
        # Updated User-Agent to be more robust
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def measure_relevance(self, page_text):
        """Calculates immediate reward (Similarity of current page to topic)."""
        with torch.no_grad():
            page_vec = self.encoder.encode(page_text, convert_to_tensor=True).to(self.device)
            similarity = torch.nn.functional.cosine_similarity(
                page_vec.unsqueeze(0), self.target_vec.unsqueeze(0)
            ).item()
        return similarity

    def get_clean_text_and_links(self, url, html):
        soup = BeautifulSoup(html, 'html.parser')
        
        # Cleanup
        for tag in soup(['script', 'style', 'nav', 'footer']):
            tag.decompose()
        
        # Get text (limit to 1000 chars for speed)
        page_text = " ".join(soup.get_text(separator=' ').split())[:1000]
        
        links = []
        for a in soup.find_all('a', href=True):
            link_text = " ".join(a.get_text().split())
            href = a['href']
            # Basic validation
            if len(link_text) < 3 or href.startswith(('#', 'javascript', 'mailto')): 
                continue
            
            # Handle relative URLs
            try:
                full_url = urljoin(url, href)
                links.append({'text': link_text, 'url': full_url})
            except:
                continue
            
        return page_text, links

    def score_links(self, page_text, links):
        if not links: return []
        
        # 1. Get All Scores
        with torch.no_grad():
            link_texts = [l['text'] for l in links]
            link_vecs = self.encoder.encode(link_texts, convert_to_tensor=True).to(self.device)
            
            # Naive Scores (Cosine)
            naive_scores = torch.nn.functional.cosine_similarity(
                link_vecs, self.target_vec.unsqueeze(0)
            ).cpu().tolist()
            if isinstance(naive_scores, float): naive_scores = [naive_scores]
            
            # RL Scores (Q-Values)
            if self.use_rl:
                n = len(links)
                page_vec = self.encoder.encode(page_text, convert_to_tensor=True).to(self.device)
                
                # Expand tensors to match batch size
                t_batch = self.target_vec.repeat(n, 1)
                p_batch = page_vec.repeat(n, 1)
                
                q_scores = self.brain(t_batch, p_batch, link_vecs).squeeze().cpu().tolist()
                if isinstance(q_scores, float): q_scores = [q_scores]
                
                # --- RANK FUSION ---
                # 1. Sort indices by Naive Score
                naive_ranked_indices = sorted(range(len(naive_scores)), key=lambda k: naive_scores[k], reverse=True)
                # 2. Sort indices by RL Score
                rl_ranked_indices = sorted(range(len(q_scores)), key=lambda k: q_scores[k], reverse=True)
                
                # 3. Create a blended score based on rank (Reciprocal Rank Fusion)
                final_scores = [0] * len(links)
                
                for rank, original_index in enumerate(naive_ranked_indices):
                    final_scores[original_index] += (1.0 / (rank + 1)) * 1.0 # Naive Weight
                
                for rank, original_index in enumerate(rl_ranked_indices):
                    final_scores[original_index] += (1.0 / (rank + 1)) * 0.5 # RL Weight
                
                return list(zip(final_scores, links))

            return list(zip(naive_scores, links))


    def crawl(self, seed_url):
        heapq.heappush(self.frontier, (-10.0, seed_url)) 
        pages_crawled = 0
        
        # Wipe old log
        with open(OUTPUT_LOG, 'w', encoding='utf-8') as f:
            f.write("")

        with open(OUTPUT_LOG, 'a', encoding='utf-8') as f:
            while self.frontier and pages_crawled < MAX_PAGES:
                neg_score, url = heapq.heappop(self.frontier)
                score = -neg_score
                
                if url in self.visited: continue
                self.visited.add(url)
                
                try:
                    response = self.session.get(url, timeout=5)
                    if response.status_code != 200: continue
                    
                    page_text, found_links = self.get_clean_text_and_links(url, response.text)
                    relevance = self.measure_relevance(page_text)
                    
                    # --- NEW: Added Timestamp for Latency Metric ---
                    log_entry = {
                        "step": pages_crawled,
                        "url": url,
                        "relevance_reward": relevance,
                        "method": "RL" if self.use_rl else "Naive",
                        "timestamp": time.time()  
                    }
                    f.write(json.dumps(log_entry) + "\n")
                    f.flush()

                    print(f"[{'RL' if self.use_rl else 'NV'}] Step {pages_crawled}: Reward={relevance:.4f} | Found {len(found_links)} links")

                    # Score and Queue new links
                    if found_links:
                        scored_candidates = self.score_links(page_text, found_links)
                        for s, link_data in scored_candidates:
                            # Dynamic Thresholding
                            if self.use_rl:
                                threshold = -999.0 # RL determines order, we trust it
                            else:
                                threshold = 0.2 # Naive filter
                            
                            if s > threshold and link_data['url'] not in self.visited:
                                heapq.heappush(self.frontier, (-s, link_data['url']))
                    
                    pages_crawled += 1
                    time.sleep(DELAY)
                    
                except Exception as e:
                    logger.warning(f"Error crawling {url}: {e}")

        # --- NEW: Save Frontier Snapshot for 'Selection Efficiency' Metric ---
        print("Saving Frontier Snapshot...")
        snapshot_data = []
        # Convert heap back to readable list (negate score back to positive)
        for neg_score, url in self.frontier:
            snapshot_data.append({
                "url": url,
                "predicted_score": -neg_score
            })
        
        with open(FRONTIER_LOG, 'w', encoding='utf-8') as f:
            json.dump(snapshot_data, f, indent=4)

        logger.info(f"Done. Logs: {OUTPUT_LOG} | Frontier: {FRONTIER_LOG}")

if __name__ == "__main__":
    bot = HybridCrawler(use_rl=USE_RL)
    bot.crawl(START_SEED)