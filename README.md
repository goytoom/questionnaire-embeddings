# Deep language models of human personality
This repository contains source codes and data for the "A Deep Language Approach to Personality Assessment" project.

# Code files
All codes are located in the `scripts` folder:
- Codes for extracting S-BERT embeddings can be found at `extract_questions_embeddings.ipynb`.
- Codes to replicate analyses and visualizations from the main publication can be found in `Study1.ipynb` to `Study4.ipynb`
- Codes to preprocess the human rater data (behavioral experiment) can be found in `preprocess_human_data.ipynb`
- Codes to replicate the target selection for the human rater studies can be found in `target_selection.ipynb`

# Data files
- Item texts, embeddings (SBERT, Word2Vec and LIWC), and original participant responses can be found under `/embeddings` in the respective questionnaire's folder.
- Human rater data (raw; Qualtrics export) can be found found under `/human_studies` in the respective questionnaire's folder.