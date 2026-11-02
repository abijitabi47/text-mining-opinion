import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.preprocessing import clean_text
from src.tag_generation import generate_tags

st.set_page_config(page_title="Opinion Mining Dashboard", page_icon="💬", layout="wide")
st.title("Opinion Mining & Tag-Based Categorization")
st.caption("Explore user reviews through preprocessing, BoW, TF-IDF, dimensionality reduction, sentiment classification, and tags.")

@st.cache_data
def sample_data():
    return pd.read_csv("data/reviews.csv")

uploaded = st.sidebar.file_uploader("Upload a CSV file (optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = sample_data()

text_candidates = [c for c in df.columns if c.lower() in ("review", "text", "comment", "feedback", "content")]
rating_candidates = [c for c in df.columns if c.lower() in ("rating", "stars", "score")]
sentiment_candidates = [c for c in df.columns if c.lower() in ("sentiment", "label", "class")]

st.sidebar.header("Column settings")
text_col = st.sidebar.selectbox("Review text column", df.columns, index=df.columns.get_loc(text_candidates[0]) if text_candidates else 0)
rating_col = st.sidebar.selectbox("Rating column (for labels)", ["None"] + list(df.columns), index=(list(df.columns).index(rating_candidates[0])+1) if rating_candidates else 0)
label_col = st.sidebar.selectbox("Existing sentiment-label column (optional)", ["None"] + list(df.columns), index=(list(df.columns).index(sentiment_candidates[0])+1) if sentiment_candidates else 0)

df = df.dropna(subset=[text_col]).copy()
df[text_col] = df[text_col].astype(str)
df["cleaned_text"] = df[text_col].apply(clean_text)
df = df[df["cleaned_text"].str.strip().ne("")].copy()

if label_col != "None":
    def normalize_label(x):
        s = str(x).strip().lower()
        if s in ("positive", "pos", "1", "4", "5"): return "Positive"
        if s in ("negative", "neg", "-1", "0", "2"): return "Negative"
        if s in ("neutral", "neu", "3"): return "Neutral"
        return None
    df["sentiment"] = df[label_col].apply(normalize_label)
elif rating_col != "None":
    def rating_to_sentiment(x):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        if x <= 2: return "Negative"
        if x == 3: return "Neutral"
        return "Positive"
    df["sentiment"] = df[rating_col].apply(rating_to_sentiment)
else:
    df["sentiment"] = None

st.info("For an honest evaluation, provide a rating column (1–5) or an existing sentiment-label column. If neither exists, the app still displays text features and tags, but will not invent sentiment labels.")
c1, c2, c3 = st.columns(3)
c1.metric("Reviews analyzed", f"{len(df):,}")
c2.metric("Unique cleaned terms", len(set(" ".join(df["cleaned_text"]).split())))
c3.metric("Sentiment labels available", int(df["sentiment"].notna().sum()))

tabs = st.tabs(["Overview & BoW", "TF-IDF", "SVD / LSA", "Sentiment", "Tags", "Review explorer"])

with tabs[0]:
    st.subheader("Dataset preview")
    st.dataframe(df[[text_col, "cleaned_text"] + (["sentiment"] if "sentiment" in df else [])].head(20), use_container_width=True)
    st.subheader("Most frequent words (Bag of Words)")
    bow = CountVectorizer(max_features=50)
    matrix = bow.fit_transform(df["cleaned_text"])
    counts = matrix.sum(axis=0).A1
    bow_df = pd.DataFrame({"term": bow.get_feature_names_out(), "count": counts}).sort_values("count", ascending=False).head(20)
    fig = px.bar(bow_df.sort_values("count"), x="count", y="term", orientation="h", title="Top terms by raw frequency")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("BoW represents each review using counts of vocabulary terms; word order is not retained.")
    st.dataframe(bow_df, use_container_width=True)

with tabs[1]:
    st.subheader("TF-IDF feature extraction")
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df["cleaned_text"])
    terms = tfidf.get_feature_names_out()
    mean_scores = X.mean(axis=0).A1
    tfidf_df = pd.DataFrame({"term": terms, "mean_tfidf": mean_scores}).sort_values("mean_tfidf", ascending=False).head(25)
    st.plotly_chart(px.bar(tfidf_df.sort_values("mean_tfidf"), x="mean_tfidf", y="term", orientation="h", title="Terms with highest average TF-IDF"), use_container_width=True)
    st.caption("TF-IDF increases the weight of terms that are frequent in a review but less common across the corpus.")
    st.dataframe(tfidf_df, use_container_width=True)
    st.download_button("Download TF-IDF summary CSV", tfidf_df.to_csv(index=False).encode("utf-8"), "tfidf_summary.csv", "text/csv")

with tabs[2]:
    st.subheader("Dimensionality reduction using Truncated SVD (LSA)")
    if min(X.shape) >= 2 and X.shape[0] >= 2:
        n_components = 2
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        coords = svd.fit_transform(X)
        plot_df = pd.DataFrame({"Component 1": coords[:, 0], "Component 2": coords[:, 1], "Review": df[text_col].str.slice(0, 100).values})
        if df["sentiment"].notna().any():
            plot_df["Sentiment"] = df["sentiment"].fillna("Unlabelled").values
            fig = px.scatter(plot_df, x="Component 1", y="Component 2", color="Sentiment", hover_data=["Review"], title="Reviews projected into two dimensions")
        else:
            fig = px.scatter(plot_df, x="Component 1", y="Component 2", hover_data=["Review"], title="Reviews projected into two dimensions")
        st.plotly_chart(fig, use_container_width=True)
        st.write("Explained variance ratio sum (two components):", round(float(svd.explained_variance_ratio_.sum()), 4))
        st.caption("Nearby points have more similar reduced text-feature patterns. The plot is exploratory, not proof of distinct groups.")
    else:
        st.warning("Not enough rows or vocabulary terms for a two-component SVD plot.")

with tabs[3]:
    st.subheader("Sentiment classification")
    labeled = df.dropna(subset=["sentiment"]).copy()
    if labeled["sentiment"].nunique() >= 2 and len(labeled) >= 6:
        counts = labeled["sentiment"].value_counts()
        st.plotly_chart(px.pie(values=counts.values, names=counts.index, title="Sentiment distribution"), use_container_width=True)
        st.write("Label rule: ratings 1–2 = Negative, 3 = Neutral, 4–5 = Positive (unless supplied labels are selected).")
        # Train/test split stratification only if each class has at least 2 examples.
        can_stratify = labeled["sentiment"].value_counts().min() >= 2
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                labeled["cleaned_text"], labeled["sentiment"], test_size=0.25,
                random_state=42, stratify=labeled["sentiment"] if can_stratify else None
            )
            model = Pipeline([("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2))),
                              ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            st.metric("Holdout accuracy", f"{accuracy_score(y_test, preds):.1%}")
            labels = sorted(labeled["sentiment"].unique())
            cm = confusion_matrix(y_test, preds, labels=labels)
            st.plotly_chart(px.imshow(cm, x=labels, y=labels, text_auto=True, labels={"x":"Predicted","y":"Actual","color":"Count"}, title="Confusion matrix"), use_container_width=True)
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)
            df["predicted_sentiment"] = model.predict(df["cleaned_text"])
            st.caption("Metrics are calculated on a held-out split of this dataset. Small or imbalanced datasets can produce unstable scores.")
        except ValueError as e:
            st.warning(f"Could not train/evaluate the classifier with this label distribution: {e}")
    else:
        st.warning("Add at least two sentiment classes and ideally several examples per class using a rating or label column to train and evaluate the classifier.")

with tabs[4]:
    st.subheader("Ranked, refined, and enriched tags")
    tags = generate_tags(df["cleaned_text"].tolist(), top_n=40)
    if tags.empty:
        st.warning("No tags could be generated from the current text.")
    else:
        st.plotly_chart(px.bar(tags.head(20).sort_values("score"), x="score", y="tag", orientation="h", title="Top ranked tags"), use_container_width=True)
        st.dataframe(tags, use_container_width=True)
        cloud_text = " ".join([((row.tag + " ") * max(1, min(10, int(row.score * 100)))) for row in tags.itertuples()])
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            wc = WordCloud(width=1200, height=500, background_color="white", collocations=False).generate(cloud_text or "user feedback")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)
        except ImportError:
            st.info("Install wordcloud to show the tag cloud; the ranked tag chart and table are available without it.")
        st.download_button("Download ranked tags CSV", tags.to_csv(index=False).encode("utf-8"), "ranked_tags.csv", "text/csv")
    st.caption("Ranking uses average TF-IDF. Refinement removes near-duplicate simple variants; enrichment adds broad, rule-based related categories where matched.")

with tabs[5]:
    st.subheader("Search and inspect reviews")
    query = st.text_input("Filter reviews by keyword")
    view = df.copy()
    if query:
        view = view[view[text_col].str.contains(query, case=False, na=False) | view["cleaned_text"].str.contains(query, case=False, na=False)]
    cols = [text_col, "cleaned_text", "sentiment"]
    if "predicted_sentiment" in df.columns:
        cols.append("predicted_sentiment")
    st.dataframe(view[cols], use_container_width=True, height=400)
    st.download_button("Download processed reviews CSV", view.to_csv(index=False).encode("utf-8"), "processed_reviews.csv", "text/csv")
