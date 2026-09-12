# codesentinel-ai

AI-powered software vulnerability detection and intelligent code review platform
using Graph Neural Networks, LLMs, and DevSecOps.

## Layout

```
backend/
├── app/        FastAPI service: AST / CFG / DFG / CPG analysis, taint and
│               rule-based vulnerability detection, risk scoring
├── ml/         CodeBERT vulnerability classifier  -> backend/ml/README.md
└── tests/
frontend/       static dashboard and analysis pages
data/           datasets, chunked data and checkpoints -> data/README.md
```

## ML module

A hierarchical [`microsoft/codebert-base`](https://huggingface.co/microsoft/codebert-base)
classifier that predicts whether a function is vulnerable (`1`) or benign (`0`),
trained on PrimeVul.

Full documentation — data pipeline, class-imbalance strategy, chunking, model
architecture, training, checkpoint/resume, threshold selection and evaluation —
is in **[backend/ml/README.md](backend/ml/README.md)**.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r backend/requirements.txt

# place primevul_{train,valid,test}.jsonl in data/raw/ first
python -m backend.ml.preprocessing.inspect_dataset
python -m backend.ml.preprocessing.validate_dataset
python -m backend.ml.preprocessing.preprocess_primevul
python -m backend.ml.preprocessing.balance_dataset
python -m backend.ml.preprocessing.chunk_dataset
python -m backend.ml.training.train
python -m backend.ml.evaluation.test_model
```

## Tests

ML module (self-contained, no dataset or GPU required):

```bash
python -m pytest backend/tests/test_ml_pipeline.py -v
```

The older service tests import `app.*` and must be run from `backend/`:

```bash
cd backend && python -m pytest tests -v
```
