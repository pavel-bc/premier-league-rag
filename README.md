# premier-league-rag

![Logo](assets/premier-league-rag.jpg)

> [!WARNING]
> Majority of the code in this repository was generated using [Claude 3.7 Sonnet](https://claude.ai)

English Premier League season simulator using:

- SQLite-based [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) (using [sqlite-vec](https://alexgarcia.xyz/blog/2024/sqlite-vec-stable-release/index.html))
- Local [Ollama](https://ollama.com) setup
- [Streamlit](https://streamlit.io) for UI

## Demo

https://github.com/user-attachments/assets/75472885-3ee1-42f6-a334-821302dde552

## Pre-requisites

```shell
$ brew install curl ollama streamlit
$ ollama pull mxbai-embed-large
$ pip install -r requirements.txt
```

## Running

Download & import historical data (takes ~10m):

```shell
$ make clean download import
```

Run simulator:

```shell
$ make run
```

## Links

- [inferablehq/sqlite-ollama-rag](https://github.com/inferablehq/sqlite-ollama-rag) - shell version for matching movies
