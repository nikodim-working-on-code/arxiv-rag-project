# 🚀 Adaptive RAG for arXiv Papers

**Self-optimizing Retrieval-Augmented Generation system** that automatically tunes ranking weights using **Bayesian Optimization** and 
**NDCG@k loss**.

---

## 🎯 Key Features

- **Multi-criteria Paper Ranking**: Combines similarity, freshness, coverage, and volume
- **Adaptive Weight Optimization**: 3-stage strategy (Random → Evolutionary → Bayesian)
- **NDCG@k Loss Function**: Measures ranking quality with DCG-based loss
- **arXiv Integration**: Fetches and ranks recent papers by category
- **PDF Full-Text Retrieval**: Downloads and extracts content from top papers

---

## 📊 System Design

### 1. **Ranking Function**

The system uses a **4-component weighted scoring model**:

$$
\text{score}(p) = \alpha \cdot s_{\text{max}} + \beta \cdot f + \gamma \cdot c_{\text{avg}} + \delta \cdot v
$$

Where each component captures a distinct relevance signal:

**Semantic Similarity** — $s_{\text{max}}$: Maximum cosine similarity between query and paper chunks

$$
s_{\text{max}} = \max_{i} \left( \frac{\mathbf{q} \cdot \mathbf{c}_i}{\|\mathbf{q}\| \|\mathbf{c}_i\|} \right)
$$

where $\mathbf{q}$ is the query embedding and $\mathbf{c}_i$ are chunk embeddings.

**Temporal Relevance** — $f$: Exponential freshness decay

$$
f = e^{-\lambda \cdot t}
$$

where $t$ is the number of days since publication, and $\lambda = 0.05$ controls decay rate.

**Coverage** — $c_{\text{avg}}$: Average similarity of chunks above threshold $\tau$

$$
c_{\text{avg}} = \frac{1}{|\{i : s_i > \tau\}|} \sum_{i : s_i > \tau} s_i
$$

This metric measures the breadth of relevant content across the paper.

**Volume** — $v$: Logarithmic ratio of relevant to total chunks

$$
v = \frac{\log(1 + n_r)}{\log(1 + n_t)}
$$

where $n_r$ is the count of relevant chunks ($s_i > \tau$) and $n_t$ is total chunks. This rewards papers with more relevant sections 
while preventing bias toward longer documents.

### 2. **Loss Function & Optimization**

The system optimizes the weight vector $\mathbf{w} = [\alpha, \beta, \gamma, \delta]$ by minimizing **ranking loss** based on **NDCG@k** 
(Normalized Discounted Cumulative Gain):

$$
\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}
$$

Where **DCG** (Discounted Cumulative Gain) is computed as:

$$
\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{r_i} - 1}{\log_2(i + 1)}
$$

Here, $r_i$ represents the relevance score at position $i$, and the logarithmic discount penalizes relevant documents appearing lower in 
the ranking.

**Loss function**:

$$
\mathcal{L}(\mathbf{w}) = 1 - \text{NDCG@k}(\mathbf{w})
$$

- $\mathcal{L} = 0$: Perfect ranking
- $\mathcal{L} = 1$: Worst possible ranking

### 3. **Adaptive Optimization Strategy**

The system employs a **3-stage adaptive approach** that transitions from exploration to exploitation:

| Stage | Query # | Algorithm | Description |
|-------|---------|-----------|-------------|
| **Baseline** | 1 | None | Establish initial performance with default weights |
| **Exploration** | 2 | Random Perturbation | Add Gaussian noise ($\sigma = 0.1$) with 30% probability |
| **Gradient Descent** | 3-5 | Evolutionary Strategy | Finite-difference gradient estimation |
| **Global Search** | 6+ | Bayesian Optimization | GP-based hyperparameter tuning |

#### Bayesian Optimization Details

For queries 6 and beyond, the system uses **Gaussian Process regression** to model the loss landscape $\mathcal{L}(\mathbf{w})$.

**Acquisition Function** — Lower Confidence Bound (LCB):

$$
\text{LCB}(\mathbf{w}) = \mu(\mathbf{w}) - \kappa \cdot \sigma(\mathbf{w})
$$

where:
- $\mu(\mathbf{w})$ is the predicted mean loss
- $\sigma(\mathbf{w})$ is the prediction uncertainty
- $\kappa = 2.0$ controls exploration vs. exploitation tradeoff

The next candidate weights are selected by:

$$
\mathbf{w}_{t+1} = \arg\min_{\mathbf{w}} \text{LCB}(\mathbf{w})
$$

The system uses a **Matérn kernel** for the GP prior, which provides smooth but non-infinitely-differentiable modeling of the loss 
function.

---

## 🏗️ Architecture

```
src/
├── config.py                    # Configuration & API keys
├── adaptive_rag.py              # Main RAG orchestrator
└── utils/
    ├── arxiv_fetcher.py         # Paper fetching with pagination
    ├── embedding_processor.py   # Embeddings & ranking logic
    └── optimizer.py             # Weight optimization algorithms
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/nikodim-working-on-code/arxiv-rag-project.git
cd adaptive-rag-arxiv

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup API Keys

Create `.env` file:

```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 3. Run Example

```bash
python3 -m examples.example_queries
```

---

## 📖 Usage

```python
from src.adaptive_rag import AdaptiveRAG

# Initialize system
rag = AdaptiveRAG()

# Run query
result = rag.run_full_pipeline(
    "What techniques reduce catastrophic forgetting in continual learning?"
)

print(f"Loss: {result['loss']:.3f}")
print(f"NDCG@10: {result['metrics']['ndcg@k']:.3f}")
print(result['answer'])
```

---

## 🧪 Optimization Performance

After 6+ queries, the system converges to stable weights with improved ranking quality:

| Metric | Initial | Optimized |
|--------|---------|-----------|
| **NDCG@10** | 0.850 | **0.942** |
| **Loss** | 0.150 | **0.058** |
| **Precision@10** | 0.800 | **1.000** |

---

## 📚 Configuration

Default weights and parameters (`src/config.py`):

```python
DEFAULT_WEIGHTS = {
    'alpha': 0.4,      # max_similarity weight
    'beta': 0.25,      # freshness weight
    'gamma': 0.25,     # coverage_avg weight
    'delta': 0.1,      # volume_bonus weight
    'tau': 0.5,        # relevance threshold
    'lambda': 0.05,    # freshness decay rate
}
```

---

## 🔬 Technical Details

### arXiv Fetching
- **Pagination**: Fetches 100 papers per request
- **Time-based filtering**: Selects N newest papers per day
- **Categories**: cs.LG, cs.AI, cs.RO, cs.CV, cs.CL

### Embeddings
- **Model**: `nomic-ai/nomic-embed-text-v1.5`
- **Chunking**: 400 chars, 80 char overlap (RecursiveCharacterTextSplitter)

### LLM
- **Provider**: Groq
- **Model**: `llama-3.3-70b-versatile`

---

## 📄 Requirements

```txt
arxiv>=2.1.0
sentence-transformers>=2.5.0
groq>=0.4.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
langchain-text-splitters>=0.0.1
PyMuPDF>=1.23.0
python-dotenv>=1.0.0
requests>=2.31.0
```

---

## 📝 License

MIT License

