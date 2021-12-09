python -m spacy init fill-config base_config.cfg config.cfg
python -m spacy debug data config.cfg
python -m spacy train config.cfg --output ./models/outcome_model