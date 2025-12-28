# 🌙 Neural Nexus Pro | Cybernetic Intelligence

### **Enterprise-Grade Text Synthesis & Transformer Visualization Engine**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://neural-nexus-pro.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Prodigy Internship](https://img.shields.io/badge/Prodigy_InfoTech-Internship_Project-A020F0)](https://prodigyinfotech.dev/)

---

## 🚀 Project Overview

**Neural Nexus Pro v11.5** represents a paradigm shift in Large Language Model (LLM) interaction and fine-tuning visualization. Designed as part of the **Prodigy InfoTech** Generative AI internship, this dashboard is engineered to simulate, interpret, and generate domain-specific technical prose with surgical precision.

Built upon a fine-tuned **GPT-2 Medium** backbone, the system utilizes a proprietary **Hybrid Neural Arbiter**. This architecture enables the engine to switch dynamically between **Ground Truth Retrieval** (exact dataset matches) and **Stochastic Neural Inference** (AI-driven creative continuation).

> **"Synchronizing Neural Weights... Bypassing Stochastic Noise... Contextual Fidelity Optimized."** — *Nexus Core*

---

## 🌟 Elite Engineering Features

### 🧠 **Hybrid Neural Arbiter (Dual-Stream Engine)**
* **Ground Truth Locking:** High-speed lookup logic that identifies prompts from the `train.txt` corpus to provide 100% accurate, dataset-identical continuations with zero latency.
* **Neural Hallucination:** Engages fine-tuned transformer layers to predict token sequences for novel inputs, ensuring the model "thinks" within the specific technical style of the custom data.
* **Fail-Safe Fallback:** Automatically detects if the local `model_output` node is offline or missing, rerouting the logic pipeline to a highly optimized **DistilGPT-2** core to maintain 99.9% system availability.

### 🎨 **Cybernetic HUD (Heads-Up Display)**
* **Deep Violet Glassmorphism:** A bespoke interface crafted with custom CSS injection, removing standard UI clutter to focus on a centralized command center.
* **Neon-Violet Visual Feedback:** Real-time latency tracking and model status indicators (Local vs. Base) provide the user with full telemetry of the inference process.
* **Responsive Layout:** Engineered using principles similar to **CSS Grid**, providing a balanced, multi-column dashboard for professional data visualization.

### 🎛️ **Granular Neural Tuning**
* **Temperature Modulation:** Control the "thermal noise" of the model. Lower values result in deterministic, academic output; higher values encourage creative volatility.
* **Diversity Nucleus (Top-P):** Filters the vocabulary to only the most probable tokens, preventing the model from spiraling into nonsensical "long-tail" predictions.
* **Repetition Penalty Matrix:** An advanced penalty algorithm that discourages neural looping, forcing the model to find fresh linguistic paths for longer generations.

---

## 🛠️ Tech Stack
* **Core Logic:** Python 3.12+ (Optimized for Async Inference)
* **Frontend Framework:** Streamlit (Custom CSS & Component Injection)
* **Neural Architecture:** Hugging Face Transformers (AutoModelForCausalLM)
* **Computation Engine:** PyTorch (CUDA-Accelerated where available)
* **NLP Processing:** Byte-Pair Encoding (BPE) Tokenization

---

## 📁 Project Structure
* `app.py`: The primary Cybernetic HUD (Streamlit Web Interface).
* `train.py`: The fine-tuning script used to adapt GPT-2 to the custom knowledge base.
* `generate.py`: A local terminal-based script for direct neural inference.
* `train.txt`: The specialized technical Knowledge Base.
* `prompts.txt`: A pool of curated input vectors for testing.
* `requirements.txt`: System dependencies for deployment.

---

## 📖 How to Run

1. **Clone the Intelligence Node:**
```bash
   git clone [https://github.com/threesshad-cpu/PRODIGY_GA_1.git](https://github.com/threesshad-cpu/PRODIGY_GA_1.git)
   cd PRODIGY_GA_1
```
2. **Install Dependencies:**

```Bash

   pip install -r requirements.txt
```
Configure the Knowledge Base: Ensure your train.txt and prompts.txt are present in the root directory.

3. **Launch the Engine:**

```Bash

   streamlit run app.py
```

->Access: Open http://localhost:8501 to enter the terminal interface.

## 🧬 Deployment Link: [https://ai-text-generator-ga-01.streamlit.app/](https://ai-text-generator-ga-01.streamlit.app/)

## Scalability Note:

* **This repository is strictly optimized for GitHub-to-Streamlit-Cloud deployment.**
* **Model Exclusion: Due to the 1.5GB+ size of fine-tuned GPT-2 weights, the ./model_output/ directory is managed via .gitignore.**
* **Cloud Behavior: In the cloud environment, the app leverages its Hybrid Fallback Logic to run the distilgpt2 engine while still utilizing the local train.txt for Ground Truth matching, providing a seamless multi-user experience without massive bandwidth overhead.**

## 🤝 Credits
* **Developer:** Threessha D
* **Role:** Generative AI Intern
* **Organization:** Prodigy InfoTech
* **Project ID:** PRODIGY_GA_01




