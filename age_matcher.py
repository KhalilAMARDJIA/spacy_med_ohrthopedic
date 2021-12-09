from query import df_pubmed
import pandas as pd
import spacy
from spacy.matcher import Matcher


query = '"Arthroplasty, Replacement, Ankle"[Mesh]'
df = df_pubmed(query= query)
df =  df.dropna(subset=['pubmed_id', 'abstract'])


nlp = spacy.load('en_core_sci_sm')
matcher = Matcher(nlp.vocab)

scores_tag = ["age", "years"]

patient_age = [
    [   
        {"OP": "?"},
        {"lower": "age"},
        {"OP": "?"},
        {"LIKE_NUM": True, "LENGTH": {">": 1}, "OP": "+"},
        {"OP": "?"},
        {"OP": "?"},
        {"OP": "?"},
        {"lower": "years"},
        {"OP": "?"},
        {"LIKE_NUM": True, "LENGTH": {">": 1}, "OP": "?"},
        {"lower": "old", "OP": "?"}
    ]
]

matcher.add("AGE_PATIENT", patient_age, greedy="LONGEST")

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
    
for age in df_match.score:
    print(age)