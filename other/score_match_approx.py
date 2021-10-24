import pandas as pd
import spacy
from spacy.matcher import Matcher

df = pd.read_csv(
    '~/Desktop/Python scripts/PUBMED/pubmed_raw_data.csv', sep=';')
df = df[df['abstract'].notna()]

nlp = spacy.load('en_core_sci_sm')
matcher = Matcher(nlp.vocab)

scores_tag = ["score", "scale", "index", "criteria"]

robust_outcome = [
    [
        {"POS": "PROPN", "IS_TITLE": True, "OP": "+"},
        {"OP": "?"},
        {"OP": "?"},
        {"OP": "?"},
        {"OP": "?"},
        {"ORTH": "("},
        {"IS_UPPER": True, "OP": "+"},
        {"OP": "?"},
        {"OP": "?"},
        {"ORTH": ")"},
        {"OP": "?"},
        {"LEMMA": {"IN": scores_tag}}
    ]
]

abbrev_outcome = [
    [
        {"ORTH": "("},
        {"IS_UPPER": True, "OP": "+"},
        {"OP": "?"},
        {"OP": "?"},
        {"ORTH": ")"},
        {"OP": "?"},
        {"LEMMA": {"IN": scores_tag}}
    ]
]

approx_outcome = [
    [
        {"IS_TITLE": True, "OP": "+"},
        {"OP": "?"},
        {"OP": "?"},
        {"OP": "?"},
        {"IS_UPPER": True, "OP": "?"},
        {"OP": "?"},
        {"IS_TITLE": True, "OP": "?"},
        {"OP": "?"},
        {"LEMMA": {"IN": scores_tag}}
    ]
]

matcher.add("SCORES_SCALES_ROBUST", robust_outcome, greedy="LONGEST")
matcher.add("SCORES_SCALES_ABBREV", abbrev_outcome, greedy="LONGEST")
matcher.add("SCORES_SCALES_APPROX", approx_outcome, greedy="LONGEST")


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


scores_list = []

for i in set(df_match.match_id):
    scores = df_match[df_match["match_id"] == i].drop_duplicates()
    scores = list(scores.score)
    scores_list.append((i,scores))
 
df["scores_found"] = ""

for i, sc_list in scores_list:
    df.loc[df['pubmed_id'] == i, 'scores_found'] = str(sc_list)

df =  df[['pubmed_id', 'doi', 'abstract', 'scores_found']]

df.to_csv('scores_identified.csv', sep= ',')