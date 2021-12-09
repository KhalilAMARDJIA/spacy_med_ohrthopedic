from query import df_pubmed
import spacy
import json
from spacy.matcher import PhraseMatcher
from tqdm import tqdm
from spacy.tokens import DocBin


query = 'Ortho AND ("Clinical Study" [Publication Type] OR "Comparative Study" [Publication Type] OR "Evaluation Study" [Publication Type] OR "Meta-Analysis" [Publication Type] OR "Multicenter Study" [Publication Type] OR "Systematic Review" [Publication Type])'
df = df_pubmed(query= query)
abstracts = df.abstract.dropna()
abstracts = abstracts.drop_duplicates()

nlp = spacy.load("en_core_sci_sm")

def db_matcher(db_path, label):
    
    with open(db_path) as file:
        db = json.load(file)  # open database containing to values to match

    phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    to_match = []

    for first_key in db.keys():
        lvl_1 = db[first_key]
        for second_key in lvl_1.keys():
            lvl_2 = lvl_1[second_key]
            for third_key in lvl_2.keys():
                lvl_3 = lvl_2[third_key]
                to_match.extend(lvl_3)

    patterns = [nlp(text) for text in to_match]

    phrase_matcher.add(label, None, *patterns)

    return phrase_matcher

outcome_matcher = db_matcher(db_path="my_db/scores_db.json", label= "OUTCOME")


def parse_train_data(list_of_text, matcher, label):
    docs = nlp.pipe(list_of_text)

    train_data = []
    for doc in docs:
        detections = ([(doc[start:end].start_char, doc[start:end].end_char, label) for idx, start, end in matcher(doc)])
        train_data.append((doc.text, {'entities': detections}))

    return train_data


training_list = (parse_train_data(list_of_text = abstracts, matcher= outcome_matcher, label= "OUTCOME"))

def rand_split_list(list, split_ratio):

    import random
    random.shuffle(list)
    split = round(len(list)*split_ratio)
    train_data = list[:split]
    test_data = list[split:]

    return train_data, test_data

train, valid = rand_split_list(list = training_list, split_ratio = 0.8)


def convert(input_list,output_path):
    nlp = spacy.blank("en") # load a new spacy model
    db = DocBin() # create a DocBin object
    TRAIN_DATA = input_list
    try:
        for text, annot in tqdm(TRAIN_DATA): # data in previous format
            doc = nlp.make_doc(text) # create doc object from text
            ents = []
            for start, end, label in annot["entities"]: # add character indexes
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    print("Skipping entity")
                else:
                    ents.append(span)
            doc.ents = ents # label the text with the ents
            db.add(doc)
    except:
        pass
    db.to_disk(output_path)

convert(input_list= train, output_path="./training_db/training_data_outcomes.spacy")
convert(input_list= valid, output_path="./training_db/validation_data_outcomes.spacy")