# Customer VOC Topic Modeling (LDA)

An unsupervised NLP pipeline built at Kimberly-Clark AI Labs to automatically surface
recurring complaint topics from customer service contact logs using LDA (Latent Dirichlet Allocation).

> **Note:** Output cells have been cleared due to company data confidentiality.
> This notebook demonstrates the methodology and code structure only.

## Overview

Customer service teams at Kimberly-Clark Korea received thousands of consultation tickets monthly
across categories such as delivery complaints, event complaints, and cancellation/refund requests.
Manual review was time-consuming and made it difficult to spot emerging issues quickly.

This pipeline applies LDA topic modeling to automatically cluster tickets into latent topics,
enabling faster identification of recurring pain points and data-driven prioritization for the CX team.

## Approach

1. **Data Loading**: Ingested internal VOC (Voice of Customer) logs covering Feb 2022 to Feb 2023
2. **Preprocessing**: Parsed ticket dates from IDs, filtered irrelevant content, removed numeric noise
3. **Korean Tokenization**: Used KoNLPy Okt morphological analyzer to extract nouns, verbs, and adjectives; filtered sparse documents
4. **Vectorization**: Applied CountVectorizer with bigram support and domain-specific stopwords
5. **LDA Modeling**: Fitted LDA with 6 topics per complaint category; tunable via `N_TOPICS`
6. **Visualization**: Used pyLDAvis for interactive topic exploration and inter-topic distance mapping
7. **Document Assignment**: Assigned each ticket to its highest-probability topic for downstream analysis
8. **Category Deep Dive**: Drilled into sub-category trends for delivery, event, and refund complaints

## Tech Stack
- Python, KoNLPy (Okt morphological analyzer)
- Scikit-learn (CountVectorizer, LatentDirichletAllocation)
- pyLDAvis (interactive topic visualization)
- Pandas, Matplotlib, Seaborn

## Key Design Decisions
- Filtered documents with fewer than 3 unique tokens to reduce noise in the LDA input
- Set `max_df=0.1` to exclude overly common terms that carry little topic signal
- Included bigrams (`ngram_range=(1,2)`) to capture compound expressions common in Korean customer feedback
- Modeled topics per L1 category rather than across all tickets to improve topic coherence

## Timeline
Kimberly-Clark AI Labs, Seoul · 2023
