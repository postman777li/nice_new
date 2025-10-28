# Hierarchical Control Legal Translation System

An agent-based high-quality legal translation system with three-layer hierarchical control: terminology, syntax, and discourse layers.

[中文文档](README_zh.md)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Database Configuration](#database-configuration)
- [Project Structure](#project-structure)
- [Development Guide](#development-guide)

## Overview

This project implements an innovative legal translation system with a three-layer agent architecture for fine-grained quality control:

1. **Terminology Layer**: Ensures accuracy and consistency of legal terms
2. **Syntax Layer**: Maintains syntactic fidelity and legal expression patterns
3. **Discourse Layer**: Preserves translation style and contextual consistency

Each layer follows an "Extract-Evaluate-Translate" workflow, achieving high-quality legal translation through agent collaboration.

## Key Features

### 🎯 Three-Layer Architecture
- **Terminology Control**: Professional term management with termbase
- **Syntax Control**: Bilingual pattern extraction and fidelity assessment
- **Discourse Control**: Translation memory retrieval and style consistency analysis

### 🤖 Agent System
Each layer contains 3 core agents:
- **Extract Agent**: Identifies key features and patterns
- **Evaluate Agent**: Analyzes quality and identifies issues
- **Translation Agent**: Improves translation based on evaluation

### 🗄️ Multi-Database Support
- **SQLite**: Terminology storage and management
- **Milvus**: Vector retrieval (terms, translation memory)
- **BM25**: Text retrieval (hybrid search strategy)

### 📊 Quality Assurance
- Real-time evaluation and feedback
- Detailed issue diagnosis
- Traceable improvement suggestions
- Complete translation trace records

## System Architecture

```
Input Text
    ↓
┌─────────────────────────────────────┐
│  Round 1: Terminology Layer          │
├─────────────────────────────────────┤
│  1. MonoExtract: Extract terms       │
│  2. Evaluate: Assess term quality    │
│  3. Translation: Generate improved   │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Round 2: Syntax Layer               │
├─────────────────────────────────────┤
│  1. BiExtract: Extract patterns      │
│  2. Evaluate: Assess syntax fidelity │
│  3. Translation: Improve syntax      │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Round 3: Discourse Layer            │
├─────────────────────────────────────┤
│  1. Query: Retrieve TM references    │
│  2. Evaluate: Analyze differences    │
│  3. Translation: Adjust style        │
└─────────────────────────────────────┘
    ↓
Final Translation
```

### Difference Analysis Example (Discourse Layer)

The discourse evaluation agent analyzes differences between current translation and historical high-quality translations:

**Terminology Differences**:
- Current uses "agreement", reference uses "contract"
- Current uses "must", reference consistently uses "shall"

**Syntax Differences**:
- Current uses active voice, reference prefers passive voice
- Current uses "if...then" conditionals, reference uses "where"

**Recommendations**:
- Suggest changing "agreement" to "contract" for consistency
- Suggest using passive voice to match reference style

## Installation

### Requirements

- Python 3.8+
- Milvus 2.3+ (for vector retrieval)
- Sufficient disk space (for translation memory and termbase)

### Installation Steps

1. **Clone Repository**
```bash
git clone <repository-url>
cd hierarchical_control_legal_transaltion_experiment
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure Milvus**

Start Milvus service (using Docker):
```bash
docker-compose up -d
```

Or refer to [Milvus Official Documentation](https://milvus.io/docs/install_standalone-docker.md)

4. **Initialize Databases**
```bash
# Create term collections
python scripts/reset_milvus_collections.py

# Import term data (optional)
python scripts/import_terms_to_db.py

# Import translation memory (optional)
python scripts/import_tm_to_db.py
```

## Quick Start

### Basic Translation

```bash
python run_translation.py \
  --source "The parties shall comply with all terms of this agreement." \
  --src-lang en \
  --tgt-lang zh \
  --verbose
```

### Batch Translation

```bash
python run_translation.py \
  --input dataset/processed/test_set_zh_en.json \
  --output results/translations.json \
  --src-lang zh \
  --tgt-lang en
```

### Configuration Options

```bash
python run_translation.py \
  --source "The party has the right to apply for administrative review." \
  --src-lang en \
  --tgt-lang zh \
  --hierarchical \          # Enable three-layer architecture (default)
  --use-termbase \          # Use termbase (default)
  --use-tm \                # Use translation memory
  --verbose                 # Show detailed output
```

## Usage Guide

### 1. Terminology Layer Workflow

The terminology layer ensures accurate translation of legal terms:

```python
from src.workflows.terminology import run_terminology_workflow

result = await run_terminology_workflow(
    orchestrator=None,
    job_id="job_001",
    config=config,
    input_text="source text"
)
```

**Output Example**:
```
📝 Extracting terms...
  Extracted 8 terms

[Details] Extracted terms:
  1. contract (合同) - confidence: 0.95
  2. party (当事人) - confidence: 0.92
  3. liability for breach (违约责任) - confidence: 0.88

🔍 Evaluating terms...
  Evaluation complete (score: 0.85)

[Details] Term evaluation:
  Accuracy: 0.90
  Consistency: 0.85
  Completeness: 0.80
```

### 2. Syntax Layer Workflow

The syntax layer ensures syntactic accuracy:

```python
from src.workflows.syntax import run_syntactic_workflow

result = await run_syntactic_workflow(
    orchestrator=None,
    job_id="job_001",
    config=config,
    input_text="round 1 result"
)
```

**Output Example**:
```
📝 Extracting syntax patterns...
  Extracted 5 patterns

[Details] Syntax patterns:
  1. shall → 应当 (modal verb, confidence: 0.95)
  2. may → 可以 (modal verb, confidence: 0.92)

🔍 Evaluating syntax...
  Evaluation complete (score: 0.78)

[Details] Syntax evaluation:
  Modality preservation: 0.85
  Connective consistency: 0.75
  Conditional logic: 0.75
  Issues: Modal verb "must" should be changed to "shall"
```

### 3. Discourse Layer Workflow

The discourse layer maintains style consistency through reference translations:

```python
from src.workflows.discourse import run_discourse_workflow

result = await run_discourse_workflow(
    orchestrator=None,
    job_id="job_001",
    config=config,
    input_text="round 2 result"
)
```

**Output Example**:
```
📝 Retrieving translation memory...
  Found 5 relevant translation memories

[Details] Translation memory:
  1. Similarity: 0.88
     Source: The parties shall perform obligations...
     Target: 双方应当按照约定履行义务...

🔍 Analyzing differences with references...
  Analysis complete (score: 0.82)

[Details] Discourse consistency analysis:
  Terminology consistency: 0.85
  Syntax consistency: 0.80
  Style consistency: 0.80
  Terminology differences: Current uses "must", reference uses "shall"
  Syntax differences: Current uses active voice, reference prefers passive
```

## Database Configuration

### Termbase (SQLite)

The termbase is located at `terms_zh_en.db`, containing:
- Term entries (source term, target term, domain, definition, etc.)
- Term relationships (synonyms, related terms, etc.)

View termbase:
```bash
python scripts/check_terms_db.py
```

Import new terms:
```bash
python scripts/import_terms_to_db.py --input terms.json
```

### Translation Memory (Milvus + BM25)

Translation memory uses hybrid retrieval:
- **Vector Retrieval (Milvus)**: Based on semantic similarity
- **BM25 Retrieval**: Based on keyword matching

Import translation memory:
```bash
python scripts/import_tm_to_db.py --input dataset/processed/train_set_zh_en.json
```

### Vector Database Collections

The system uses the following Milvus collections:
- `legal_terms_zh_en`: Chinese-English term vectors
- `legal_tm_zh_en`: Chinese-English translation memory
- `legal_tm_zh_ja`: Chinese-Japanese translation memory

Reset collections:
```bash
python scripts/reset_milvus_collections.py
```

## Project Structure

```
hierarchical_control_legal_transaltion_experiment/
├── run_translation.py              # Main translation script
├── run_experiment.py               # Batch experiment script
├── requirements.txt                # Dependencies
├── configs/
│   └── default.yaml                # Default configuration
├── src/
│   ├── agents/                     # Agent modules
│   │   ├── terminology/            # Terminology layer agents
│   │   │   ├── mono_extract.py    # Monolingual term extraction
│   │   │   ├── evaluate.py        # Term evaluation
│   │   │   └── translation.py     # Term translation
│   │   ├── syntax/                 # Syntax layer agents
│   │   │   ├── bi_extract.py      # Bilingual syntax extraction
│   │   │   ├── syntax_evaluate.py # Syntax evaluation
│   │   │   └── syntax_translation.py # Syntax translation
│   │   └── discourse/              # Discourse layer agents
│   │       ├── discourse_query.py  # TM query
│   │       ├── discourse_evaluate.py # Consistency analysis
│   │       └── discourse_translation.py # Discourse integration
│   ├── lib/                        # Core libraries
│   │   ├── llm_client.py          # LLM client
│   │   ├── vector_db.py           # Vector database
│   │   ├── term_db.py             # Termbase
│   │   └── tm_db.py               # Translation memory
│   └── workflows/                  # Workflows
│       ├── terminology.py          # Terminology workflow
│       ├── syntax.py               # Syntax workflow
│       └── discourse.py            # Discourse workflow
├── scripts/                        # Utility scripts
│   ├── import_terms_to_db.py      # Import terms
│   ├── import_tm_to_db.py         # Import TM
│   ├── bi_term_extract.py         # Bilingual term extraction
│   └── reset_milvus_collections.py # Reset Milvus
├── dataset/                        # Datasets
│   └── processed/                  # Processed data
│       ├── train_set_zh_en.json   # Training set
│       └── test_set_zh_en.json    # Test set
└── docs/                           # Documentation
    ├── README_terminology_import.md
    └── README_bilingual_extract.md
```

## Development Guide

### Adding New Agents

1. Create a new agent file in the appropriate layer directory
2. Inherit from `BaseAgent` class
3. Implement the `execute()` method
4. Export in `__init__.py`

Example:
```python
from ..base import BaseAgent, AgentConfig

class MyAgent(BaseAgent):
    def __init__(self, locale: str = 'zh'):
        super().__init__(AgentConfig(
            name='my:agent',
            role='my_role',
            domain='terminology',
            specialty='My specialty',
            quality='review',
            locale=locale
        ))
    
    async def execute(self, input_data: Dict[str, Any], ctx: Optional[AgentRunContext] = None):
        # Implement agent logic
        pass
```

### Modifying Workflows

Workflows are defined in `src/workflows/` directory. When modifying workflows, ensure:

1. Consistent input/output formats
2. Comprehensive error handling
3. Clear logging output
4. Support for verbose mode

### Custom Evaluation Metrics

Evaluation metrics are defined in each evaluate agent. Customize by modifying prompts or adding new evaluation dimensions.

### Testing

Run tests:
```bash
# Test single translation
python run_translation.py --source "test text" --src-lang zh --tgt-lang en

# Test batch processing
python test/test_batch_processing.py

# Test data import
python test/test_import.py
```

## Configuration

### Environment Variables

Create a `.env` file and set the following variables:

```bash
# LLM API Configuration
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-4

# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```

### Configuration File

Configure system parameters in `configs/default.yaml`:

```yaml
translation:
  hierarchical: true        # Enable hierarchical architecture
  use_termbase: true        # Use termbase
  use_tm: false            # Use translation memory
  use_rules: false         # Use rule table
  
models:
  llm_model: "gpt-4"
  embedding_model: "text-embedding-3-small"
  
database:
  milvus_host: "localhost"
  milvus_port: 19530
```

## FAQ

### Q: How to improve translation quality?

A: 
1. Ensure termbase data is complete
2. Import more high-quality translation memories
3. Check detailed evaluation results in verbose mode
4. Adjust configuration based on evaluation suggestions

### Q: How to add new language pairs?

A:
1. Prepare termbase for the language pair
2. Prepare translation memory for the language pair
3. Add corresponding language identifiers in code
4. Import data to Milvus

### Q: System is running slowly?

A:
1. Check Milvus service status
2. Reduce top_k parameter value
3. Consider using faster embedding models
4. Enable caching mechanisms

### Q: How to debug agents?

A:
1. Use `--verbose` flag for detailed output
2. Check log files
3. Add breakpoints in agent code
4. Use trace information to track execution flow

## License

[To be added]

## Contributing

Issues and Pull Requests are welcome!

## Citation

If you use this project in your research, please cite:

```bibtex
[To be added]
```

## Contact

[To be added]

---

**Note**: This project is under active development and APIs may change.
