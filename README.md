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

## Attributions
- Dataset: _Ciobanu, Alexandru (2026), “PolyglotFakeFacts: A multilingual dataset of fake and real news across politics, security, and social domains_v2.0”, Mendeley Data, V1, doi: 10.17632/8yfrm6z9dx.1_
