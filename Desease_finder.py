import spacy
import pandas as pd
from query import df_pubmed
from plotly import express as px


data = df_pubmed('Akin Osteotomy')
data = data[data['abstract'].notna()]
text = ''

for abstract in data.abstract:
    text += abstract

nlp = spacy.load("en_ner_bc5cdr_md")

# Named entity recognition
 
doc = nlp(text)

ner_data = {
    'entity':[],
    'label':[]
}

for ent in doc.ents:
    ner_data['entity'].append(ent.text)
    ner_data['label'].append(ent.label_)

ner_data = pd.DataFrame(ner_data)
ner_table = pd.crosstab(ner_data['entity'], ner_data['label']).reset_index()

display_n = 30
ner_table = ner_table.sort_values('DISEASE', ascending= False).head(display_n).head(display_n).sort_values('DISEASE', ascending= True)

fig = px.bar(
    template = 'simple_white',
    x = 'DISEASE',
    y = 'entity', 
    color_discrete_sequence=px.colors.diverging.curl,
    data_frame=ner_table,
    title=f'PubMed data extracted from abstracts'
    )

fig.update_traces(marker_line_color='black',marker_line_width=1)
fig.update_layout(font_family="Courier New")
fig.show()
