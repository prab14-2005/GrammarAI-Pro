# Grammar Error Detection and Correction Using Natural Language Processing

A professional, full-stack Natural Language Processing (NLP) micro-project for detecting grammatical, spelling, punctuation, and stylistic errors in English sentences and providing contextual corrections and morphological linguistic analysis.

## GrammarAI Pro upgrade

GrammarAI Pro preserves the original Flask, NLTK, LanguageTool, custom-rule, SQLite history, and browser interface features. It adds an optional **AI Correct** action backed by the public Hugging Face model [`prithivida/grammar_error_correcter_v1`](https://huggingface.co/prithivida/grammar_error_correcter_v1): a T5/PyTorch `text2text-generation` model. It is loaded lazily on the first AI request and is configurable through `GRAMMAR_AI_MODEL`; no API key is required.

The model download is substantial and model output is not guaranteed to preserve every nuance. The app labels this output as generative and keeps it separate from the existing rule-based correction result. If the model or its dependencies cannot load, `/api/ai-correct` returns a safe 503 message while `/api/check` continues to work.

### AI API

`POST /api/ai-correct` with `{"text":"She go to college every day."}` returns the original text, model-generated corrected text, model identifier, concise disclosure, and the existing NLTK analysis. Empty input, malformed JSON, and text over 5,000 characters return 400.

### Model setup and real inference

```bash
copy .env.example .env
pip install -r requirements.txt
python -c "import nltk; [nltk.download(x) for x in ['punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng','wordnet','omw-1.4']]"
python app.py
```

Click **AI Correct** with `She go to college every day.`. The first request downloads and initializes the model; later requests reuse it. For a controlled failure check, set `GRAMMAR_AI_MODEL=not/a-real-model` and restart: the AI endpoint should return 503, while the standard checker still works.

### Deployment

`render.yaml` prepares a one-worker Flask deployment, but verify the current free-tier RAM/disk limits before deploying: T5/PyTorch model downloads and cold starts can exceed small free instances. SQLite is ephemeral on common free/serverless hosts, so deployed history may reset on restart. A Hugging Face Space with an appropriate persistent-storage plan is generally preferable for this model, but requires account login and manual setup. Do not use the included Vercel configuration for real model inference because serverless cold starts and ephemeral storage make it unsuitable.

### ISE-2 report and demo outline

1. Problem: rule-only grammar checking misses contextual cases.
2. Architecture: browser → Flask APIs → NLTK/LanguageTool/custom rules and optional T5 model → SQLite.
3. Demonstrate rule checking, POS/lemma analysis, then **AI Correct** and disclose its limitations.
4. Show history and the graceful unavailable-model message.
5. Discuss privacy, bias, semantic drift, model size/cold starts, and future evaluation with a labeled GEC dataset.

---

## 📌 Project Overview

This project implements an end-to-end NLP pipeline for automated English grammatical error detection and correction (GEC). Designed for academic and practical demonstrations, the application combines **rule-based computational linguistics** (such as Subject-Verb Agreement, Subject-Auxiliary Inversion, and Determiner Agreement) with **statistical language modeling** (via LanguageTool) and **morpho-syntactic feature extraction** (via NLTK and WordNet).

The project features a responsive modern web dashboard, real-time error highlighting with interactive suggestions, academic POS/lemmatization inspection, and persistent SQLite telemetry.

---

## 🚀 Key Features

1. **Grammar Checker & Correction Engine**:
   - Detects grammatical inconsistencies, verb agreement errors, tense issues, missing auxiliary inversion, and typos.
   - Generates fully corrected sentences/paragraphs with original whitespace and casing preserved.
   - Provides 1-click **Copy to Clipboard**, **Check Again**, and **Clear** controls.
2. **Interactive Color-Coded Error Highlighting**:
   - Errors are highlighted directly inside the sentence with distinct visual categories:
     - 🟠 **Grammar Errors**: Subject-Verb Agreement, verb forms, question inversion.
     - 🔴 **Spelling Errors**: Typos and misspelled words.
     - 🔵 **Punctuation Errors**: Spacing, capitalization, terminal punctuation.
     - 🟣 **Style / Other**: Duplications and stylistic improvements.
   - Clicking any error opens an interactive **Error Inspector** showing the incorrect phrase, NLP rule explanation, and 1-click **Apply Fix** button.
3. **Statistical & Quality Metrics**:
   - Word count and sentence count.
   - Total errors detected and corrections suggested.
   - **Grammar Accuracy Percentage**: Calculated via $\max\left(0, 100 - \frac{\text{errors}}{\text{words}} \times 100\right)\%$.
4. **Academic NLP Analysis Inspector**:
   - **Pipeline Flow Visualizer**: Raw text $\rightarrow$ Preprocessing $\rightarrow$ Tokenization $\rightarrow$ POS Tagging $\rightarrow$ Lemmatization $\rightarrow$ Rule Processing $\rightarrow$ Output.
   - **Penn Treebank POS Tagging**: Each token tagged with its syntactic part of speech (e.g., `NNP`, `VBZ`, `PRP`, `WRB`) and plain-English explanation.
   - **WordNet Lemmatization**: Displays the dictionary root form (lemma) for every word.
   - **Syntactic & Lexical Metrics**: Lexical diversity percentage, unique words, and average sentence length.
5. **Persistent History (SQLite)**:
   - Every grammar check is automatically saved with original text, corrected text, error count, and timestamp.
   - Search/filter previous checks.
   - Detailed modal view.
   - Delete individual items or clear all history.
6. **Modern Responsive UI**:
   - Dark mode & Light mode toggle (persisted in `localStorage`).
   - Clean typography with `Plus Jakarta Sans` and `JetBrains Mono`.
   - Accessible, mobile-friendly card layout.
   - 1-click academic benchmark test cases.

---

## 🛠️ Technology Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python 3.10+, Flask 3.1, SQLite 3, Flask-CORS |
| **NLP & Linguistics** | NLTK 3.10, WordNet, LanguageTool (`language_tool_python` / REST API) |
| **Frontend** | HTML5 (Semantic), Vanilla CSS3 (Custom Design System, Dark/Light theme), JavaScript (ES6+) |
| **Testing** | Python `unittest` suite |

---

## 🔄 Project Architecture & Workflow

```
User Enters Text
       │
       ▼
Text Preprocessing (Cleaning & Normalization)
       │
       ▼
Sentence Tokenization (NLTK sent_tokenize)
       │
       ▼
Word Tokenization (NLTK word_tokenize)
       │
       ▼
Part-of-Speech Tagging (NLTK pos_tag - Penn Treebank)
       │
       ▼
Morphological Lemmatization (WordNet Lemmatizer)
       │
       ▼
Syntactic Rule Engine (SVA, Question Inversion, Articles, Duplication)
       │
       ▼
Statistical Language Verification (LanguageTool Engine)
       │
       ▼
Deduplication & Conflict Resolution
       │
       ▼
Correction Generation & Interactive Highlighting
       │
       ▼
SQLite Database Persistence & Dashboard Telemetry
```

---

## 📂 Project Directory Structure

```
grammar-error-detector/
│
├── app.py                      # Flask backend application & REST API routes
├── requirements.txt            # Python package dependencies
├── README.md                   # Complete documentation & run guide
├── database.db                 # SQLite database (auto-created on startup)
│
├── nlp/                        # Core Natural Language Processing modules
│   ├── __init__.py             # NLP package exports
│   ├── preprocessing.py        # Text cleaning, normalization, contractions
│   ├── analyzer.py             # Sentence/word tokenization, POS tagging, lemmatization
│   └── grammar_checker.py      # Rule engine + LanguageTool integration & correction
│
├── database/                   # Database operations
│   ├── __init__.py             # Database package exports
│   └── db.py                   # SQLite tables, CRUD operations, telemetry stats
│
├── templates/
│   └── index.html              # Modern, responsive single-page application
│
├── static/
│   ├── css/
│   │   └── style.css           # Custom CSS styling (dark/light mode, cards, badges)
│   └── js/
│       └── script.js           # Client-side state, API calls, interactive highlights
│
└── tests/
    └── test_grammar.py         # Unit tests covering all benchmark test cases
```

---

## ⚡ Installation and Setup Guide

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed on your system.
- Git (optional, for cloning).
- VS Code or any text editor.

### 2. Open Project in VS Code
Open VS Code, select **File > Open Folder...**, and select the `grammar-error-detector` directory:
```bash
cd grammar-error-detector
```

### 3. Create a Virtual Environment (Recommended)
In the VS Code terminal (PowerShell or Bash):
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download Required NLTK Corpora
Run the following command to download the tokenizer, POS tagger, and WordNet lexical database:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## ▶️ Running the Application

### Start the Flask Server
Run the application using:
```bash
python app.py
```

You should see output similar to:
```
=======================================================
 Grammar Error Detector & Corrector NLP Project
 Running at: http://127.0.0.1:5000
=======================================================
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

### Access the Web Interface
Open your web browser and visit:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🧪 Verified Benchmark Test Cases

The application includes built-in verification for common grammatical problems:

| Test Case | Input Sentence | Expected & Corrected Output | Error Category |
|---|---|---|---|
| **Test Case 1** | `She go to school every day.` | `She goes to school every day.` | Subject-Verb Agreement |
| **Test Case 2** | `He don't like apples.` | `He doesn't like apples.` | SVA (Negative Contraction) |
| **Test Case 3** | `I has a book.` | `I have a book.` | Auxiliary Agreement |
| **Test Case 4** | `They was playing cricket.` | `They were playing cricket.` | Plural Past-Tense Agreement |
| **Test Case 5** | `Where you are going?` | `Where are you going?` | Subject-Auxiliary Inversion |
| **Extra Case 6** | `She go to college every day.` | `She goes to college every day.` | Subject-Verb Agreement |
| **Extra Case 7** | `I want a apple and an book.` | `I want an apple and a book.` | Indefinite Article Rule |
| **Extra Case 8** | `He is a good studnet.` | `He is a good student.` | Spelling Correction |
| **Extra Case 9** | `This is the the best day.` | `This is the best day.` | Duplicate Word Removal |

---

## 🧪 Running the Unit Tests

Execute the automated test suite with Python's built-in `unittest` runner:
```bash
python -m unittest discover -s tests
```

Expected result:
```
...........
----------------------------------------------------------------------
Ran 11 tests in ~5.8s

OK
```

---

## 📡 RESTful API Documentation

| Method | Endpoint | Description | Request Body | Response |
|---|---|---|---|---|
| `POST` | `/api/check` | Main grammar check and correction | `{"text": "string"}` | Full report with original, corrected, errors, highlights, statistics, and NLP breakdown |
| `POST` | `/api/analyze` | Dedicated linguistic token breakdown | `{"text": "string"}` | Sentences, tokens, POS tags, lemmas, lexical diversity |
| `GET` | `/api/history` | Retrieve check history items | Optional `?limit=50` | `{"success": true, "history": [...]}` |
| `GET` | `/api/history/<id>`| Retrieve single check item by ID | None | `{"success": true, "item": {...}}` |
| `DELETE`| `/api/history/<id>`| Delete individual check item | None | `{"success": true, "message": "..."}` |
| `POST` | `/api/clear-history`| Wipe all check history | None | `{"success": true, "message": "..."}` |
| `GET` | `/api/stats` | Aggregated dashboard telemetry | None | Total checks, errors, corrections, avg accuracy, category counts |

---

## 🎓 Academic Concepts Explained

1. **Tokenization**:
   - The process of segmenting continuous text into discrete units (tokens).
   - Sentence tokenization splits running text into sentences using boundary markers (`.`, `!`, `?`).
   - Word tokenization decomposes each sentence into individual words and punctuation tokens.
2. **Part-of-Speech (POS) Tagging**:
   - Syntactic categorization of each token according to its grammatical role in the sentence (e.g., Noun, Verb, Pronoun, Adjective).
   - Uses the Penn Treebank tagset (e.g., `VBZ` = 3rd person singular present verb, `PRP` = personal pronoun).
3. **Lemmatization**:
   - Algorithmic reduction of inflected words to their canonical dictionary root (lemma).
   - Unlike stemming (which crudely truncates suffixes), lemmatization considers the word's POS tag (e.g., *"playing"* tagged as a verb $\rightarrow$ *"play"*; *"was"* tagged as a verb $\rightarrow$ *"be"*).
4. **Subject-Verb Agreement (SVA)**:
   - A syntactic constraint in English requiring the grammatical number and person of the subject to match the finite verb (e.g., singular subject *"She"* requires singular verb form *"goes"*, not base form *"go"*).
5. **Subject-Auxiliary Inversion**:
   - A syntactic rule in direct English questions requiring the auxiliary verb to precede the subject pronoun (e.g., *"Where you are going?"* $\rightarrow$ *"Where are you going?"*).

---

## 🔮 Future Improvements

- Integrate fine-tuned Transformer-based sequence-to-sequence models (e.g., Google T5 / BART) for complex discourse-level paraphrasing.
- Support for multilingual grammar checking (Spanish, French, German).
- Real-time keystroke grammar checking with debounced WebSocket connections.
- Export corrections report to PDF or DOCX format.
