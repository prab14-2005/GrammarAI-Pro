# GrammarAI Pro — ISE-2 report outline

## 1. Title and objective
GrammarAI Pro: Generative AI-Based Grammar Error Detection and Correction. The objective is to retain a transparent NLP grammar checker while adding a configurable pretrained sequence-to-sequence correction model.

## 2. Existing system
The inherited Flask app preprocesses English text, uses NLTK tokenization/POS tagging/WordNet lemmatization, applies custom agreement/article/duplication rules, queries LanguageTool when available, and stores checks in SQLite.

## 3. Generative enhancement
The new `/api/ai-correct` endpoint uses the public Hugging Face model `prithivida/grammar_error_correcter_v1` through Transformers and PyTorch. It is T5-based text-to-text generation, has no required API key, is configured by `GRAMMAR_AI_MODEL`, and loads only on the first AI request. It is not a fallback for the rule engine.

## 4. Architecture and data flow
Browser → Flask → validation → (a) existing NLTK/rules/LanguageTool or (b) lazy-loaded T5 correction → JSON response → SQLite history. The database adds optional `ai_metadata_json` through an in-place migration, preserving existing history rows.

## 5. Tests actually executed
`python -m unittest discover -s tests -v`: 14 passed. Rule API checks returned 200 for: `She go to college every day.`, `I has a book.`, `I want a apple.`, `This is the the best project.`, and `She studies Natural Language Processing.` Empty and 10,001-character requests returned 400. `/api/analyze`, `/api/history`, and `/api/stats` returned 200. AI failure testing returned 503 because this environment does not have Transformers installed. Model output was mocked only in unit tests; no real model output is claimed.

## 6. Limitations and ethics
Generative output may miss errors or change meaning; users must review it. Model downloads, RAM, CPU latency, cold starts, and ephemeral SQLite storage limit free deployment. Do not send confidential writing to third-party services unless their privacy terms are acceptable. Evaluate performance on a labeled grammar-correction dataset before making accuracy claims.

## 7. Five-slide presentation
1. Problem and goals.
2. Original NLP pipeline.
3. GrammarAI Pro architecture and selected T5 model.
4. Live rule-based and AI correction demonstration.
5. Results, limitations, deployment considerations, and future work.

## 8. Three-to-five-minute demonstration
Open the app and run `She go to college every day.` with **Check Grammar**. Show the highlighted rule, correction, and NLP table. Then use **AI Correct** on a multi-sentence paragraph and compare original versus generated result, explaining that it must be reviewed. Open History to show persistence. Finally, show a missing-model 503 message or explain the first download, proving that the standard checker remains available when AI is unavailable.
