# KAVER: Knowledge-Augmented Verifiable Chain-of-Thought Prompting for Dialogue Answer Generation

## Overview

Knowledge-grounded dialogue systems struggle to maintain factual consistency across multi-turn conversations, frequently generating responses that misalign with or contradict external knowledge sources. These failures often arise from the lack of explicit mechanisms to structure reasoning and verify claims against external knowledge. Existing approaches either inject knowledge implicitly, losing interpretability, or rely on end-to-end fine-tuning, which is computationally expensive and prone to inconsistencies. We propose KAVER, a Knowledge-Augmented Verifiable Reasoning framework that addresses these limitations through a decoupled architecture separating symbolic reasoning from neural generation. 
KAVER consists of three tightly coupled components: context-adaptive subgraph construction via k-hop expansion with relevance filtering; multi-path symbolic reasoning through bidirectional graph traversal that yields structured reasoning traces; and structure-aware response synthesis with entity-level self-consistency aggregation grounded in knowledge graph evidence.KAVER trains only lightweight LoRA adapters (<1% of parameters) while keeping the backbone LLM frozen, enabling parameter-efficient deployment. Experiments on three benchmarks show that KAVER achieves the highest Entity F1 (77.2%) and MoverScore (75.7) on CamRest, and the highest BLEU (19.7) and Combined Score (104.2) on MultiWOZ 2.1. While retrieval-focused methods achieve higher entity coverage on larger-scale datasets, KAVER provides explicit reasoning chains that establish traceability from generated responses to knowledge graph evidence. 

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT (for LoRA adapter training)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Data Preparation

Download the datasets and place them in `data/`:
- **CamRest**: Single-domain restaurant recommendation
- **MultiWOZ 2.1**: Multi-domain task-oriented dialogue
- **SMD (InCar)**: Navigation, weather, and calendar domains

### 2. Model Setup

Download the Qwen1.5-7B base model and configure its path in `main.py` and `run_training.py`.

### 3. Training

**Stage 1 – Train the graph reasoning module:**
```bash
python run_training.py --dataset camrest
```
Supported dataset values: `camrest`, `incar`, `WOZ2_1`

**Stage 2 – Train domain LoRA adapters:**
```bash
python main.py --dataset camrest
```

### 4. Evaluation

```bash
python main.py --dataset camrest   # CamRest
python main.py --dataset woz2_1    # MultiWOZ 2.1
python main.py --dataset incar     # SMD (InCar)
```


## Main Results

| Dataset       | BLEU     | Entity F1    | MoverScore   | Combined     |
|---------------|----------|--------------|--------------|--------------|
| CamRest       | 26.1     | 77.2     | 75.7     | –            |
| MultiWOZ 2.1  | 19.7 | 51.0         | 65            | 104.2   |
| InCar (SMD)   | 22.0     | 69.0         | 71.8     | –            |



## Project Structure

```
.
├── data/                          # Dataset files and loaders
│   ├── camrest/
│   ├── incar/
│   ├── woz2_1/
│   └── dataset.py
│
├── scripts/               # Dataset preprocessing scripts
│   ├── dataset_camrest.py
│   ├── dataset_incar.py
│   └── dataset_woz2_1.py
│
├── KG_CoT_Model/                  # Core symbolic reasoning modules
│   ├── model.py                   # Multi-hop graph reasoning model
│   └── produce.py                 # Path reconstruction and chain generation
│
├── generation/                    # Response generation components
│   ├── enhanced_generator.py      # LLM-based response generator
│   ├── prompt_engineering.py      # Chain-of-Thought prompt construction
│
│
│├── self_consistency/              # Self-consistency inference mechanisms
│   └── self_consistency_integration.py
│
├── training/                      # Training utilities
│   ├── adapter_finetuner.py       # LoRA domain adapter training
│
├── utils/                         # Utility modules
│   ├── kg_utils.py
│   ├── dialogue_state.py
│   └── ...
│
├── main.py                        # Main entry point
├── run_training.py                # Training entry point
└── README.md
```

## Citation

If you use KAVER in your research, please cite the associated paper.
