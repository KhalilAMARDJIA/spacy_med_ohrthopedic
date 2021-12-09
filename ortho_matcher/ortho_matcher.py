import pandas as pd
import spacy
from spacy.matcher import Matcher
import json

df = pd.read_csv("pubmed_raw.csv", sep = '|')
df = df[df['abstract'].notna()]
nlp = spacy.load('en_core_sci_sm')
matcher = Matcher(nlp.vocab)

with open ("ortho_matcher/patterns.json") as file:
    patterns = json.load(file)

patient_age = [patterns[0]['age']]
sample_size = [patterns[0]['sample_size']]
outcomes = [patterns[0]['outcomes']]

matcher.add("AGE_PATIENT", patient_age, greedy="LONGEST")
matcher.add("SAMPLE_SIZE", sample_size, greedy="LONGEST")
matcher.add("OUTCOME", outcomes, greedy="LONGEST")

abstracts_w_id = []  # create a tupple of (abstract, id) to trace

for abstract,pubmed_id in zip(df.abstract,df.pubmed_id): 
    abstracts_w_id.append((abstract,pubmed_id))


tags = []
scores = []
matched_ids = []

for doc, abstract_id in list(nlp.pipe(abstracts_w_id, as_tuples=True)):
    matches = matcher(doc)

    for match in matches:
        for match_id, start, end in matches:

            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]  # The matched span
            tags.append(string_id)
            scores.append(span.text.replace('\n', ''))
            matched_ids.append(abstract_id)


df_match = pd.DataFrame({
    'match_id': matched_ids,
    'score': scores,
    'tag': tags})

df_match = df_match.drop_duplicates()
