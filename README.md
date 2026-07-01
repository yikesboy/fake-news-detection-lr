# fake-news-detection-logistic-regression

## Run

Get required tooling if nix is installed.

```sh
nix develop
```

Install the Python dependencies:

```sh
uv sync
```

Convert the raw excel data and run the pipeline:

```sh
uv run xlsx-to-jsonl
uv run main.py
```

## Results

The current training pipeline uses the English translated article text as the
only feature. Text is transformed with `TfidfVectorizer` and classified with
`LogisticRegression`. Hyperparameters are selected with a 5-fold stratified
`GridSearchCV` using macro F1 as the scoring metric, followed by threshold
tuning with `TunedThresholdClassifierCV`.

The best final configuration was:

- `classifier__C`: `30.0`
- `classifier__class_weight`: `balanced`
- `features__translated_article_tfidf__min_df`: `20`
- `features__translated_article_tfidf__ngram_range`: `(1, 2)`
- `features__translated_article_tfidf__sublinear_tf`: `True`
- tuned decision threshold: `0.4242`

Final evaluation on the predefined test split:

| Metric | Score |
| --- | ---: |
| Grid-search macro F1 | 0.9796 |
| Threshold CV macro F1 | 0.9807 |
| Test accuracy | 0.9857 |
| Test macro F1 | 0.9856 |
| Test weighted F1 | 0.9857 |

Per-class test results:

| Class | Precision | Recall | F1-score | Support |
| --- | ---: | ---: | ---: | ---: |
| real | 0.9885 | 0.9838 | 0.9862 | 1052 |
| fake | 0.9826 | 0.9876 | 0.9851 | 971 |

Confusion matrix:

```text
[[1035   17]
 [  12  959]]
```

## Attributions
- Dataset: _Ciobanu, Alexandru (2026), “PolyglotFakeFacts: A multilingual dataset of fake and real news across politics, security, and social domains_v2.0”, Mendeley Data, V1, doi: 10.17632/8yfrm6z9dx.1_
