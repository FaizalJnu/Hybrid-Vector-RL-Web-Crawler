import json
import numpy as np
import matplotlib.pyplot as plt
import os
from groq import Groq  # <--- NEW IMPORT
from deployment import USE_RL
# --- CONFIGURATION ---

VISITED_LOG = f"crawl_results_{'RL' if USE_RL else 'NAIVE'}.jsonl"
FRONTIER_LOG = f"frontier_snapshot_{'RL' if USE_RL else 'NAIVE'}.json"

# Thresholds
RELEVANCE_THRESHOLD = 0.35  
HIGH_VALUE_THRESHOLD = 0.75 

# GROQ Configuration
# Replace this with actual key or set it as an env variable
GROQ_API_KEY = "gsk.." 
GROQ_MODEL = "llama-3.3-70b-versatile"

def llm_judge_trajectory(visited_pages, goal="Deep Learning Syllabus"):
    """
    Sends the crawl path to Llama 3 (via Groq) to evaluate logic.
    """
    if not GROQ_API_KEY or "gsk_" not in GROQ_API_KEY:
        print("Missing Groq API Key. Skipping LLM Judge.")
        return "N/A (No API Key)"
    
    # Construct a clean path summary (First 20 + Last 10 to see start & finish)
    path_str = "--- START OF CRAWL ---\n"
    
    # Take a slice to fit context window easily
    display_pages = visited_pages[:20] + visited_pages[-10:] if len(visited_pages) > 30 else visited_pages
    
    for i, p in enumerate(display_pages):
        step_num = p.get('step', i)
        path_str += f"Step {step_num}: {p.get('url')} (Reward: {p.get('relevance_reward'):.2f})\n"
    
    path_str += "--- END OF CRAWL ---"

    prompt = f"""
    You are a judge evaluating a Web Crawler's intelligence.
    GOAL: Find the "{goal}".
    
    CRAWL PATH:
    {path_str}
    
    TASK:
    1. Analyze the URLs. Does the crawler seem to be "tunneling" logically (e.g., Home -> Dept -> Course)?
    2. Or is it clicking randomly/greedily?
    3. Note: A low reward doesn't mean bad logic (it might be a necessary navigation step).
    
    OUTPUT FORMAT:
    Logic Score: [0-100]
    Verdict: [1 sentence justification]
    """
    
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a strict technical judge evaluating AI agents."},
                {"role": "user", "content": prompt}
            ],
            model=GROQ_MODEL,
            temperature=0.1, # Keep it deterministic
            max_tokens=150
        )
        return chat_completion.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Groq Error: {str(e)}"

def main():
    print("--- GENERATING THESIS METRICS (GROQ POWERED) ---")

    if not os.path.exists(VISITED_LOG):
        print(f"Error: {VISITED_LOG} not found.")
        return

    # 1. Load Visited Data
    visited_pages = []
    with open(VISITED_LOG, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                if 'relevance_reward' in data:
                    visited_pages.append(data)
            except json.JSONDecodeError:
                continue

    # Sort by step to ensure timeline is correct
    visited_pages.sort(key=lambda x: x.get('step', 0))
    
    if not visited_pages:
        print("No valid visited pages found.")
        return

    # 2. Load Frontier Data
    unvisited_links = []
    if os.path.exists(FRONTIER_LOG):
        try:
            with open(FRONTIER_LOG, 'r', encoding='utf-8') as f:
                unvisited_links = json.load(f)
        except:
            unvisited_links = []

    # --- METRIC 1: HARVEST RATE ---
    total_visited = len(visited_pages)
    relevant_visited = sum(1 for p in visited_pages if float(p['relevance_reward']) > RELEVANCE_THRESHOLD)
    harvest_rate = (relevant_visited / total_visited) * 100 if total_visited > 0 else 0

    # --- METRIC 2: PEAK REWARD ---
    rewards = [float(p['relevance_reward']) for p in visited_pages]
    peak_reward = max(rewards) if rewards else 0
    success_flag = "YES" if peak_reward > HIGH_VALUE_THRESHOLD else "NO"

    # --- METRIC 3: TRAJECTORY GAIN ---
    first_10_avg = np.mean(rewards[:10]) if len(rewards) >= 10 else np.mean(rewards)
    last_10_avg = np.mean(rewards[-10:]) if len(rewards) >= 10 else np.mean(rewards)
    traj_gain = last_10_avg - first_10_avg

    # --- METRIC 4: SELECTION EFFICIENCY ---
    if unvisited_links:
        frontier_scores = [l.get('predicted_score', 0) for l in unvisited_links]
        frontier_mean = np.mean(frontier_scores)
        # Count how many items in queue are better than the average
        high_quality_queue = sum(1 for s in frontier_scores if s > frontier_mean)
        selection_efficiency = (high_quality_queue / len(unvisited_links)) * 100
    else:
        selection_efficiency = 0

    # --- METRIC 5: LATENCY ---
    avg_latency = 0.0
    if 'timestamp' in visited_pages[0] and 'timestamp' in visited_pages[-1]:
        total_duration = visited_pages[-1]['timestamp'] - visited_pages[0]['timestamp']
        if total_visited > 1:
            avg_latency = (total_duration / total_visited) * 1000 # to ms

    # --- METRIC 6: LLM JUDGEMENT (GROQ) ---
    print("\nSending crawl path to Groq (Llama 3)...")
    llm_verdict = llm_judge_trajectory(visited_pages)

    # --- REPORT GENERATION ---
    print("\n" + "="*70)
    print(f"       THESIS METRICS REPORT (QUALITY V2)       ")
    print("="*70)
    print(f"{'METRIC':<25} | {'GOAL':<15} | {'RESULT':<10}")
    print("-" * 70)
    print(f"{'Harvest Rate':<25} | {'> 30%':<15} | {harvest_rate:.2f}%")
    print(f"{'Peak Reward':<25} | {'> 0.75':<15} | {peak_reward:.4f} ({success_flag})")
    print(f"{'Trajectory Gain':<25} | {'Positive':<15} | {traj_gain:+.4f}")
    print(f"{'Selection Efficiency':<25} | {'> 50%':<15} | {selection_efficiency:.2f}%")
    print(f"{'Avg Latency':<25} | {'< 1000ms':<15} | {avg_latency:.2f} ms")
    print("-" * 70)
    print(f"LLM VERDICT:\n{llm_verdict}")
    print("="*70)

    # --- PLOTTING ---
    plt.figure(figsize=(10, 5))
    steps = [p.get('step', i) for i, p in enumerate(visited_pages)]
    
    plt.plot(steps, rewards, marker='o', alpha=0.3, color='gray', label='Step Reward')
    
    # Trendline
    window = 5
    if len(rewards) > window:
        trend = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(steps[window-1:], trend, color='blue', linewidth=2, label=f'Trend ({window}-step)')

    plt.axhline(y=RELEVANCE_THRESHOLD, color='r', linestyle='--', label='Relevance Threshold')
    plt.axhline(y=HIGH_VALUE_THRESHOLD, color='g', linestyle='--', label='Success Threshold')

    method = visited_pages[0].get('method', 'Unknown')
    plt.title(f"Search Trajectory: {method}")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_plot = f"results_groq_{method}.png"
    plt.savefig(output_plot)
    print(f"\nPlot saved to '{output_plot}'")

if __name__ == "__main__":
    main()