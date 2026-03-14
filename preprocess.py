import json
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# --- CONFIG ---
DATA_PATH = "crawled_data.jsonl"
OUTPUT_PATH = "vectorized_data.pt"
MODEL_NAME = "all-MiniLM-L6-v2"

def main():
    print(f"--- 1. Loading Encoder: {MODEL_NAME} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(MODEL_NAME).to(device)
    
    print(f"--- 2. Processing {DATA_PATH} ---")
    # We will store a dictionary: { "url_string": { "page_vec": tensor, "links": [ {"text_vec": tensor, "url": str} ] } }
    vector_db = {}
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Found {len(lines)} pages. Encoding... (This may take a minute)")
    
    for line in tqdm(lines):
        try:
            data = json.loads(line)
            url = data['url']
            
            # 1. Encode Page Content (Context)
            # We take the first 500 chars to save time/space, usually contains the meat.
            page_text = data.get('page_text_content', '')[:500] 
            page_vec = encoder.encode(page_text, convert_to_tensor=True, device=device)
            
            # 2. Encode Links (Action Space)
            # We process links in a batch for massive speedup
            links_data = data.get('outgoing_links', [])
            valid_links = []
            link_texts_batch = []
            
            for l in links_data:
                # Basic cleaning check
                if l.get('text') and len(l['text']) > 2:
                    link_texts_batch.append(l['text'])
                    valid_links.append(l) # Keep metadata for the environment logic
            
            if link_texts_batch:
                # Batch Encode! This is 100x faster than doing it one by one
                link_vecs = encoder.encode(link_texts_batch, convert_to_tensor=True, device=device)
                
                # Structure the processed links
                processed_links = []
                for i, link_meta in enumerate(valid_links):
                    processed_links.append({
                        'url': link_meta['url'],
                        'vector': link_vecs[i].cpu() # Move to CPU RAM to save VRAM for training
                    })
            else:
                processed_links = []

            # Store in DB
            vector_db[url] = {
                'page_vec': page_vec.cpu(), # Move to CPU RAM
                'links': processed_links
            }
            
        except json.JSONDecodeError:
            continue

    print(f"--- 3. Saving to {OUTPUT_PATH} ---")
    torch.save(vector_db, OUTPUT_PATH)
    print("Done! You can now run the fast training script.")

if __name__ == "__main__":
    main()