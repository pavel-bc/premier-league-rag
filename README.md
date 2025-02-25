# ollama-sqlite-rag

```
⚠️ Warning: Most of the code in this repository is generated using Claude 3.7 Sonnet
```

SQLite-based Premier League simulator using [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) based local [Ollama](https://ollama.com) & [Streamlit](https://streamlit.io) setup.

## Pre-requisites

```shell
$ brew install ollama sqlite sqlite-utils streamlit
$ ollama pull mxbai-embed-large
$ sqlite-utils install sqlite-utils-sqlite-vec
$ pip install -r requirements.txt
```

## Running

```shell
$ streamlit run streamlit_app.py
```

## Links

- https://github.com/inferablehq/sqlite-ollama-rag