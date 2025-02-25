install:
	brew install ollama sqlite sqlite-utils streamlit
	ollama pull mxbai-embed-large
	sqlite-utils install sqlite-utils-sqlite-vec
	pip install -r requirements.txt

run:
	streamlit run streamlit_app.py

.PHONY: install run