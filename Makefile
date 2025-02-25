KAGGLE_URL = https://www.kaggle.com/api/v1/datasets/download/evangower/premier-league-matches-19922022
DESTINATION_FILE = premier-league-matches.zip

install:
	brew install ollama sqlite streamlit
	ollama pull mxbai-embed-large
	pip install --upgrade -r requirements.txt

run:
	streamlit run streamlit_app.py

download:
	curl -L -o $(DESTINATION_FILE) $(KAGGLE_URL)
	unzip $(DESTINATION_FILE)
	rm -rf $(DESTINATION_FILE)

import:
	python import_data.py

clean:
	rm -rf premier-league-matches.zip
	rm -rf premier-league-matches.csv
	rm -rf premier_league.db

.PHONY: download import install run clean