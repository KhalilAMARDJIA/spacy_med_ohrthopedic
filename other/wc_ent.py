import pandas as pd
import spacy
nlp = spacy.load('en_core_sci_lg')

from wordcloud import WordCloud, ImageColorGenerator
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
df = pd.read_csv(
    '~/Desktop/Python scripts/PUBMED/pubmed_raw_data.csv', sep=';')
abstracts = df.abstract.dropna()
docs = []
for text in abstracts:
    docs.append(nlp(text))


ents = []
for doc in docs:
    for ent in list(set(doc.ents)):
        ents.append(ent.text)

df = pd.DataFrame({'key': ents})


stops = ["patient", "patients", "subject", "case", "cases",  "search",
         "research", "literature", "used", "using", "analysis",
         "showed", "clinical", "studies", "study", "age",
         "underwent", "outcome", "surgery", "aim", "day", "result",
         "results", "outcomes", "performed", "total", "background",
         "procedure", "years", "months", "score", "average",
         "mean", "with", "without", "may", "range", "follow",
         "up", "significantly", "significant", "abstract",
         "however", "can", "p", "n", "mm", "one", "method",
         "included", "procedure", "treatment", "month", "year",
         "follow", "performed", "group", "groups","follow-up"]



for stop in stops:
    df = df[df['key'] != stop]

ents_count = df['key'].value_counts().to_dict()

mask = np.array(Image.open("./plots/masks/mask.png"))


wc = WordCloud(
    background_color="white",
    mode="RGBA",
    mask=mask,
    width=4000, height=4000,
    max_words=100, 
    prefer_horizontal=0.9,
    relative_scaling= 0.9
)
wc.generate_from_frequencies(frequencies=ents_count)

# create colored mask
image_colors = ImageColorGenerator(mask)

# creat plot
plt.figure(figsize=[10,10])
plt.imshow(wc.recolor(color_func= image_colors), interpolation="bilinear")
plt.axis("off")
plt.savefig('wordcloud_keywords.pdf', dpi=600)