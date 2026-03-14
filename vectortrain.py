import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
from collections import deque

# --- CONFIGURATION ---
DATA_PATH = "crawled_data.jsonl"
SEEDS_PATH = "seeds.json"
# CHECKPOINT_PATH = "checkpoint_generalized.pth"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
VECTOR_DB_PATH = "vectorized_data.pt"
CHECKPOINT_PATH = "checkpoint_advanced.pth"
TARGET_TOPIC = "EuroLeague Basketball News"

# --- Hyperparameters ---
MAX_STEPS = 500_000 
BATCH_SIZE = 2048  
REPLAY_BUFFER_SIZE = 100_000
START_TRAINING_AFTER = 10_000

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Stores a transition.
        state: Dict {'target': str, 'current_text': str, 'links': list}
        action: str (The text of the link clicked)
        reward: float
        next_state: Dict (same structure as state)
        done: bool
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Returns a batch of transitions.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# --- 1. THE ADVANCED MODEL ---
class AdvancedQNetwork(nn.Module):
    def __init__(self, embedding_dim=384):
        super(AdvancedQNetwork, self).__init__()
        
        # We process State (Goal+Page) and Action (Link) separately first
        # This is a "Siamese" style processing before merging
        self.state_fc = nn.Linear(embedding_dim * 2, 512)
        self.action_fc = nn.Linear(embedding_dim, 512)
        
        # Merged Pathway (ResNet Style with LayerNorm)
        self.fc1 = nn.Linear(1024, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, target_vec, page_vec, link_vec):
        # 1. State Processing
        state_x = torch.cat([target_vec, page_vec], dim=1)
        state_emb = self.relu(self.state_fc(state_x))
        
        # 2. Action Processing
        action_emb = self.relu(self.action_fc(link_vec))
        
        # 3. Combine
        x = torch.cat([state_emb, action_emb], dim=1)
        
        # 4. Deep Network with Residuals (Optional, kept simple here for stability)
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.relu(self.fc3(x))
        
        return self.out(x)

# --- 2. FAST IN-MEMORY ENVIRONMENT ---
class FastVectorEnv(gym.Env):
    def __init__(self, db_path, target_text):
        print("Loading Vector DB into RAM...")
        self.db = torch.load(db_path)
        self.urls = list(self.db.keys())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Pre-compute Target Vector ONCE
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2").to(self.device)
        self.target_vec = encoder.encode(target_text, convert_to_tensor=True)
        print("Environment Ready.")

    def reset(self):
        # Pick random start
        self.current_url = random.choice(self.urls)
        self.data = self.db[self.current_url]
        self.steps = 0
        self.visited = set()
        self.previous_relevance = 0.0
        return self._get_obs()

    def _get_obs(self):
        # Return TENSORS, not text. Zero CPU overhead.
        # Format: (Page_Vector, List_of_Link_Vectors)
        return self.data['page_vec'], self.data['links']

    def step(self, action_idx):
        self.steps += 1
        links = self.data['links']
        
        # Termination conditions
        if self.steps > 20: return self._get_obs(), 0, True, {}
        if action_idx >= len(links): return self._get_obs(), -1.0, False, {}
        
        # Get Action info
        link_obj = links[action_idx]
        next_url = link_obj['url']
        link_vec = link_obj['vector']
        
        # Calculate Reward (Pure Math, no text parsing)
        # Cosine Similarity between Link Vector and Target Vector
        # We do this manually for speed: (A . B) / (|A| * |B|)
        # Since sentence_transformers are normalized, just Dot Product!
        similarity = torch.dot(link_vec.to(self.device), self.target_vec).item()
        
        # Reward Logic
        shaping = (similarity - self.previous_relevance) * 10
        reward = shaping - 0.05
        done = False
        
        if similarity > 0.75: # Threshold for "Good Match"
            reward += 5.0
            done = True
        
        # Move state
        if next_url in self.db:
            self.current_url = next_url
            self.data = self.db[next_url]
            self.visited.add(next_url)
        else:
            # Dead end (URL not in our crawled dataset)
            reward -= 0.1
            # Stay on same page
        
        self.previous_relevance = similarity
        return self._get_obs(), reward, done, {}

# --- 3. TRAINING LOOP (PATCHED) ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with Hindsight Experience Replay (HER)...")
    
    # Load Environment
    env = FastVectorEnv(VECTOR_DB_PATH, TARGET_TOPIC) 
    
    # Model Setup
    q_net = AdvancedQNetwork().to(device)
    optimizer = optim.AdamW(q_net.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    # Replay Buffer
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    epsilon = 1.0
    
    page_vec, links = env.reset()
    total_reward = 0
    
    # Tracking
    loss_val = 0
    
    for step in range(MAX_STEPS):
        # --- 1. ROBUST ACTION SELECTION ---
        if not links:
            # DEAD END: We must reset or step blindly to a new random state
            # In this Env, let's just force a reset to keep training moving
            page_vec, links = env.reset()
            continue # Skip this step

        # Epsilon Greedy
        if random.random() < epsilon:
            action_idx = random.randint(0, len(links)-1)
        else:
            with torch.no_grad():
                # Batch Inference for Speed
                candidates = torch.stack([l['vector'] for l in links]).to(device)
                n_links = len(candidates)
                
                t_vecs = env.target_vec.repeat(n_links, 1)
                p_vecs = page_vec.to(device).repeat(n_links, 1)
                
                q_vals = q_net(t_vecs, p_vecs, candidates)
                action_idx = torch.argmax(q_vals).item()

        # --- 2. STEP ---
        (next_page_vec, next_links), reward, done, _ = env.step(action_idx)
        total_reward += reward
        
        # Capture the link vector we actually clicked
        clicked_link_vec = links[action_idx]['vector']
        
        # Store transition
        # We store CPU tensors to save GPU VRAM for the batch
        replay_buffer.append((
            page_vec.cpu(), 
            clicked_link_vec.cpu(), 
            reward, 
            next_page_vec.cpu(), 
            done
        ))
        
        # --- 3. TRAINING (HER ENABLED) ---
        if len(replay_buffer) > START_TRAINING_AFTER:
            batch = random.sample(replay_buffer, BATCH_SIZE)
            
            # Unpack and move to GPU
            b_page = torch.stack([x[0] for x in batch]).to(device)
            b_link = torch.stack([x[1] for x in batch]).to(device)
            b_rew  = torch.tensor([x[2] for x in batch], dtype=torch.float32).to(device).unsqueeze(1)
            b_next = torch.stack([x[3] for x in batch]).to(device) # Next pages
            
            # A. STANDARD LEARNING (Did we find the original target?)
            original_targets = env.target_vec.repeat(BATCH_SIZE, 1)
            
            # B. HINDSIGHT LEARNING (Pretend the next page WAS the target)
            # This teaches the agent the semantic relationship: Link -> Page
            her_targets = b_next.clone() # The goal is exactly what we found
            her_rewards = torch.full((BATCH_SIZE, 1), 5.0).to(device) # MAX REWARD
            
            # Combine batches (2x Size)
            # Input: Target + Page + Link
            cat_targets = torch.cat([original_targets, her_targets], dim=0)
            cat_pages   = torch.cat([b_page, b_page], dim=0)
            cat_links   = torch.cat([b_link, b_link], dim=0)
            cat_rewards = torch.cat([b_rew, her_rewards], dim=0)
            
            # Optimization
            curr_q = q_net(cat_targets, cat_pages, cat_links)
            loss = loss_fn(curr_q, cat_rewards)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()

            # Decay Epsilon
            if epsilon > 0.05: 
                epsilon *= 0.99995

        # --- 4. RESET IF DONE ---
        if done:
            if step % 100 == 0:
                print(f"Step {step} | Reward: {total_reward:.2f} | Eps: {epsilon:.3f} | Loss: {loss_val:.4f}")
            page_vec, links = env.reset()
            total_reward = 0
        else:
            page_vec = next_page_vec
            links = next_links

    # Save Final Brain
    torch.save(q_net.state_dict(), CHECKPOINT_PATH)
    print("Training Complete. Saved checkpoint.")

if __name__ == "__main__":
    main()