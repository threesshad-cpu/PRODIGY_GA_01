import streamlit as st
import random
import time
import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NEURAL NEXUS PRO",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;800&family=Inter:wght@400;700&display=swap');
    
    :root { 
        --neon: #A020F0; 
        --deep: #4B0082; 
        --bg: #030305; 
        --card: #0E0E15; 
        --text: #E0E0E0;
    }
    
    .stApp { background-color: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; }
    
    /* HIDE SIDEBAR & MENU */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    section[data-testid="stSidebar"] {display: none;}
    
    /* VIOLET SLIDERS */
    div[data-baseweb="slider"] > div > div { background-color: var(--neon) !important; }
    div[role="slider"] { background-color: var(--neon) !important; border: 2px solid #fff !important; box-shadow: 0 0 10px var(--neon); }
    
    /* NEON BUTTONS */
    .stButton>button {
        background: linear-gradient(90deg, var(--deep), var(--neon));
        color: white; border: none; font-family: 'Orbitron'; font-weight: bold;
        transition: 0.3s; width: 100%; height: 50px; text-transform: uppercase; letter-spacing: 1px;
        margin-top: 10px;
    }
    .stButton>button:hover { box-shadow: 0 0 25px var(--neon); transform: translateY(-2px); }
    
    /* CARDS */
    .control-card {
        background: var(--card); border: 1px solid var(--deep); border-top: 4px solid var(--neon);
        border-radius: 8px; padding: 25px; box-shadow: 0 0 15px rgba(160, 32, 240, 0.15);
        height: 100%;
        position: relative;
    }
    
    /* BOLD DROPDOWN */
    .stSelectbox div[data-baseweb="select"] {
        background: #000 !important; border: 1px solid var(--neon) !important;
        color: white !important; font-weight: 800 !important; font-size: 1.1rem;
    }
    
    /* READ-ONLY PROMPT BOX */
    .prompt-display {
        background: #000; border: 1px dashed var(--deep);
        color: #AAA; padding: 15px; font-family: 'Inter'; font-style: italic;
        border-radius: 6px; margin-bottom: 15px;
    }
    
    /* OUTPUT CONTAINERS */
    .output-container {
        background: #000; border: 1px solid var(--neon); border-radius: 8px;
        padding: 20px; margin-top: 15px; position: relative;
    }
    .output-text {
        font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 700;
        color: #fff; line-height: 1.6;
    }
    .highlight { color: var(--neon); text-shadow: 0 0 8px var(--deep); }
    .badge {
        font-size: 0.7rem; padding: 3px 8px; border-radius: 4px; 
        font-family: 'Orbitron'; margin-bottom: 8px; display: inline-block; font-weight: bold;
    }
    
    /* HEADINGS */
    h1 { font-family: 'Orbitron'; font-weight: 900; letter-spacing: 3px; margin: 0; padding: 0; }
    h3 { font-family: 'Orbitron'; color: var(--neon); margin-top: 0; margin-bottom: 20px; font-size: 1.2rem; letter-spacing: 1px; }
    
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.8rem; }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. FILE LOADING (SEPARATE PROMPTS & TRAIN FILES)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_files():
    """
    Loads PROMPTS for the dropdown.
    Loads TRAINING DATA for the Ground Truth lookup.
    """
    prompts = []
    ground_truth_lines = []

    # 1. Load Prompts
    if os.path.exists("prompts.txt"):
        with open("prompts.txt", "r", encoding="utf-8", errors="replace") as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
    else:
        prompts = ["Error: prompts.txt not found."]

    # 2. Load Training Data (Ground Truth)
    if os.path.exists("train.txt"):
        with open("train.txt", "r", encoding="utf-8", errors="replace") as f:
            ground_truth_lines = [line.strip() for line in f.readlines() if line.strip()]
    
    return prompts, ground_truth_lines

# -----------------------------------------------------------------------------
# 3. AI ENGINE LOADING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_engine():
    """Loads Model. Prioritizes local folder."""
    model_path = "./model_output"
    
    # Check Local
    if os.path.exists(model_path):
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            model = GPT2LMHeadModel.from_pretrained(model_path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            return tokenizer, model, True, f"Local Model ({device})"
        except:
            pass

    # Fallback
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, False, "Base Model (Fallback)"

# -----------------------------------------------------------------------------
# 4. LOGIC: SENTENCE COMPLETION & HYBRID GENERATION
# -----------------------------------------------------------------------------
def complete_sentence(text):
    """Trims text to the last punctuation mark."""
    last_punc = max(text.rfind('.'), text.rfind('?'), text.rfind('!'))
    if last_punc != -1:
        return text[:last_punc+1]
    return text + "..."

def generate_hybrid_logic(prompt, temp, top_p, penalty, max_new_tokens, num_samples, gt_lines):
    clean_prompt = prompt.strip()
    results = []
    
    # --- 1. GROUND TRUTH (Exact Match) ---
    match = None
    for line in gt_lines:
        if clean_prompt.lower() in line.lower():
            match = line
            break
            
    if match:
        formatted = match.replace(clean_prompt, f"<span class='highlight'>{clean_prompt}</span>")
        if "<span" not in formatted:
             formatted = f"<span class='highlight'>{clean_prompt}</span> " + match[len(clean_prompt):]
        results.append({"type": "GROUND TRUTH (DATASET)", "text": formatted})
    
    # --- 2. AI GENERATION ---
    needed = num_samples - len(results)
    
    if needed > 0:
        tokenizer, model, is_local, _ = load_engine()
        inputs = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=top_p,
                repetition_penalty=penalty,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=needed 
            )
        
        for i, out in enumerate(outputs):
            raw_text = tokenizer.decode(out, skip_special_tokens=True)
            polished_text = complete_sentence(raw_text)
            hl_text = polished_text.replace(prompt, f"<span class='highlight'>{prompt}</span>")
            results.append({"type": f"AI GENERATION {i+1}", "text": hl_text})
                
    return results

# -----------------------------------------------------------------------------
# 5. MAIN UI LAYOUT
# -----------------------------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🎡NEURAL NEXUS <span style='color:#A020F0'>PRO</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#888; letter-spacing:3px; margin-bottom:20px;'>CUSTOM TRAINING VISUALIZER</p>", unsafe_allow_html=True)

col_hud, col_main = st.columns([1, 2.5], gap="large")
 
# --- LEFT: HUD CONTROLS ---
with col_hud:
    # Combined container + Header as requested
    st.markdown("""
        <div class='control-card'>
            <h3>SYSTEM CONTROLS</h3>
    """, unsafe_allow_html=True)
    
    temp = st.slider("TEMPERATURE", 0.1, 1.5, 0.7, 0.1)
    top_p = st.slider("DIVERSITY (TOP-P)", 0.5, 1.0, 0.9, 0.05)
    penalty = st.slider("REP. PENALTY", 1.0, 2.0, 1.2, 0.1)
    max_new = st.slider("MAX NEW TOKENS", 20, 200, 50, 10)
    
    st.markdown("<hr style='border-color:#4B0082; margin: 15px 0;'>", unsafe_allow_html=True)
    
    num_samples = st.slider("TOTAL SAMPLES", 1, 5, 3)
    
    # Status
    _, _, is_local, path_info = load_engine()
    st.markdown(f"<div style='margin-top:15px; font-size:0.7rem; color:#666;'>Model: {path_info}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- RIGHT: INTERFACE ---
with col_main:
    # Load Data
    prompts_list, gt_lines = load_files()
    
    # Random shuffle
    if 'display_prompts' not in st.session_state:
        safe_sample = min(15, len(prompts_list))
        st.session_state.display_prompts = random.sample(prompts_list, safe_sample)

    c_sel, c_ref = st.columns([6, 1])
    with c_sel:
        # Prompt Selection
        sel_prompt = st.selectbox("", st.session_state.display_prompts, label_visibility="collapsed")
    with c_ref:
        if st.button("🔄", help="Shuffle Prompts"):
            safe_sample = min(15, len(prompts_list))
            st.session_state.display_prompts = random.sample(prompts_list, safe_sample)
            st.rerun()

    # Determine Incomplete Fragment (Logic-only, no edit box)
    # Use full line from prompts.txt as the "Incomplete Prompt" directly
    # OR cut it if the user wants it to be incomplete. 
    # Assuming prompts.txt contains STARTING phrases:
    final_input_prompt = sel_prompt 

    # Read-only visual feedback instead of editable text area
    st.markdown(f"<div class='prompt-display'>Active Vector: \"{final_input_prompt}...\"</div>", unsafe_allow_html=True)

    if st.button("INITIATE NEURAL INFERENCE"):
        start = time.time()
        st.session_state.results = generate_hybrid_logic(
            final_input_prompt, temp, top_p, penalty, max_new, num_samples, gt_lines
        )
        st.session_state.lat = time.time() - start

    # OUTPUTS
    if 'results' in st.session_state:
        st.markdown(f"<p style='text-align:right; color:#888; font-size:0.8rem; margin-top:5px;'>Latency: {st.session_state.lat:.2f}s</p>", unsafe_allow_html=True)
        
        for item in st.session_state.results:
            is_gt = "GROUND TRUTH" in item['type']
            color = "#00FF00" if is_gt else "#A020F0"
            
            st.markdown(f"""
            <div class='output-container' style='border-color: {color};'>
                <div class='badge' style='background-color: {color}; color:black;'>{item['type']}</div>
                <div class='output-text'>
                {item['text']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='control-card' style='text-align:center; opacity:0.6; padding:40px; margin-top:20px; border-style:dashed;'>
            <h4 style='color:#A020F0; font-family:Orbitron'>AWAITING INPUT</h4>
            <p>Select a prompt to verify Dataset Match vs AI Generation.</p>
        </div>
        """, unsafe_allow_html=True)