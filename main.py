from typing import Tuple, Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET_FILE_PATH = "data/processed/news.jsonl"

NGRAM_RANGE_TRANSLATED_ARTICLE: Tuple[int, int] = (1,2)
NGRAM_RANGE_ORIGINAL_ARTICLE: Tuple[int, int] = (3,5)
MIN_DOCUMENT_FRQUENCY = 2
SUBLINEAR_TF = True

TRANSLATED_ARTICLE_COL = "english_translated_version"
#ORIGINAL_ARTICLE_COL = "news_original_text"
#HEADLINE_COL = "news_headline"
LABEL_COL = "label"

FEATURE_COLUMNS = [
    TRANSLATED_ARTICLE_COL,
    #ORIGINAL_ARTICLE_COL,
    #HEADLINE_COL
]

REQUIRED_COLS = [LABEL_COL, *FEATURE_COLUMNS]


df = pd.read_json(DATASET_FILE_PATH, lines=True)

#print(df.shape)
#print(df.columns.tolist())
#print(df.head())

df[REQUIRED_COLS] = df[REQUIRED_COLS].replace(
    r"^\s*$",
    pd.NA,
    regex=True,
)

label_mapping = {
    "real": 0,
    "fake": 1,
}

df[LABEL_COL] = (
    df[LABEL_COL]
    .astype(str)
    .map(label_mapping)
)

df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)

pipeline = Pipeline([
    ("features", ColumnTransformer(
        [
            (
                "translated_article_tfidf",
                TfidfVectorizer(
                    ngram_range=NGRAM_RANGE_TRANSLATED_ARTICLE,
                    min_df=MIN_DOCUMENT_FRQUENCY,
                    sublinear_tf=SUBLINEAR_TF
                ),
                TRANSLATED_ARTICLE_COL,
            ),
            #(
            #    "original_article_tfidf",
            #    TfidfVectorizer(
            #        analyzer="char_wb",
            #        ngram_range=NGRAM_RANGE_ORIGINAL_ARTICLE,
            #        min_df=MIN_DOCUMENT_FRQUENCY,
            #        sublinear_tf=SUBLINEAR_TF
            #    ),
            #    ORIGINAL_ARTICLE_COL,
            #),
            #(
            #    "headline_tfidf",
            #    TfidfVectorizer(
            #        analyzer="char_wb",
            #        ngram_range=NGRAM_RANGE_ORIGINAL_ARTICLE,
            #        min_df=MIN_DOCUMENT_FRQUENCY,
            #        sublinear_tf=SUBLINEAR_TF,
            #    ),
            #    HEADLINE_COL,
            #)
        ]
    )),
    ("classifier", LogisticRegression(max_iter=2000))
])

X = df[FEATURE_COLUMNS]
y = df[LABEL_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

report = classification_report(
    y_true=y_test,
    y_pred=y_pred,
    target_names=["real", "fake"],
    digits=4,
)

matrix = confusion_matrix(
    y_true=y_test, 
    y_pred=y_pred
)

print(report)
print(matrix)