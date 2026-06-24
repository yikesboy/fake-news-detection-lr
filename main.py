from typing import Final, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TRAIN_SPLIT_DATASET_FILE_PATH = "data/processed/train.jsonl"
TEST_SPLIT_DATASET_FILE_PATH = "data/processed/test.jsonl"

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


train_df = pd.read_json(TRAIN_SPLIT_DATASET_FILE_PATH, lines=True)
test_df = pd.read_json(TEST_SPLIT_DATASET_FILE_PATH, lines=True)

#print(train_df.shape)
#print(train_df.columns.tolist())
#print(train_df.head())

for df in (train_df, test_df):
    df[REQUIRED_COLS] = df[REQUIRED_COLS].replace(
        r"^\s*$",
        pd.NA,
        regex=True,
    )

label_mapping = {
    "real": 0,
    "fake": 1,
}

for df in (train_df, test_df):
    df[LABEL_COL] = (
        df[LABEL_COL]
        .astype(str)
        .map(label_mapping)
    )

train_df = train_df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
test_df = test_df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)

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

X_train_full = train_df[FEATURE_COLUMNS]
y_train_full = train_df[LABEL_COL]
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[LABEL_COL]

X_train, X_validation, y_train, y_validation = train_test_split( X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

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