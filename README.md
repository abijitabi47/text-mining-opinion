# Opinion Mining & Tag-Based Categorization Dashboard

A Streamlit case-study starter implementing text preprocessing, Bag of Words, TF-IDF, Truncated SVD/LSA, sentiment classification, and tag ranking/refinement/enrichment.

## 1. Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2. Dataset

`data/reviews.csv` contains a small illustrative headphone-review dataset with 1–5 ratings. It is included so the application runs immediately. Replace it with a suitable public review dataset before using results as findings in your report.

The app supports CSV uploads. Select the review text column and either a rating column or an existing sentiment-label column in the sidebar.

## 3. Sentiment mapping

For star ratings, the starter uses:
- 1–2: Negative
- 3: Neutral
- 4–5: Positive

The app trains a TF-IDF + Logistic Regression classifier and reports holdout metrics only when sufficient labeled data are available. With the small sample file, treat metrics as a demonstration, not a reliable estimate of real-world performance.

## 4. Method overview

1. Clean text: lowercase, remove URLs/punctuation/numbers, tokenize, remove English stop words, and apply conservative suffix normalization.
2. BoW: `CountVectorizer` counts terms.
3. TF-IDF: `TfidfVectorizer` creates weighted unigram/bigram features.
4. Dimensionality reduction: `TruncatedSVD` projects TF-IDF vectors into two dimensions.
5. Sentiment: Logistic Regression on labeled review text.
6. Tags: average TF-IDF ranking, simple duplicate refinement, and rule-based related-tag enrichment.

## 5. Report screenshots

Capture screenshots of:
- Dataset preview and BoW chart
- TF-IDF top-term chart and table
- SVD/LSA scatter plot
- Sentiment distribution, confusion matrix, and metrics
- Ranked tags chart and word cloud
- Review explorer

Explain what each screenshot shows. Do not report invented metrics; record the values produced by your own run.

## 6. Dataset reference

Example public source to investigate for a larger dataset:
https://huggingface.co/datasets/fancyzhx/amazon_polarity

Note: Amazon Polarity is a binary polarity dataset; it does not directly provide neutral labels. For this assignment's three-class requirement, prefer a publicly documented star-rating dataset or clearly document how neutral examples are obtained.

## 7. Limitations

- The included data is small and illustrative.
- The text normalizer uses lightweight suffix rules, not a full linguistic lemmatizer.
- Enrichment rules are transparent keyword-based rules, not an external knowledge graph.
- SVD plots are exploratory and can lose information.
- Classification metrics depend on dataset size, class balance, and label quality.
