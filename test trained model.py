import spacy
from query import df_pubmed
nlp = spacy.load('models/outcome_model/model-best')

query = 'proximal interphalangeal arthrodesis'

df = df_pubmed(query= query)
abstracts = df.abstract.dropna()
abstracts = abstracts.drop_duplicates()


docs = nlp.pipe(abstracts)


for doc in docs:
    print(list(doc.ents))