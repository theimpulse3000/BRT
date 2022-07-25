# compatible with spacy version : 2.1.0
from __future__ import unicode_literals
from __future__ import print_function
# import requirements
import random
import logging
from pathlib import Path
import re
import json
import spacy
import plac
# reference - https://medium.com/@dataturks/automatic-summarization-of-resumes-with-ner-8b97a5f562b
# For more details, see the documentation:
# Training: https://spacy.io/usage/training
# NER: https://spacy.io/usage/linguistic-features#named-entities

# Removes leading and trailing white spaces from entity spans
def entity_span_trim(data: list) -> list:
    span_tokens_invalid = re.compile(r'\s')
    cleaned_data = []
    for text, annotations in data:
        enti = annotations['entities']
        valid_entities = []
        for start, end, label in enti:
            valid_start = start
            valid_end = end
            while valid_start < len(text) and span_tokens_invalid.match(text[valid_start]):
                valid_start += 1
            while valid_end > 1 and span_tokens_invalid.match(text[valid_end - 1]):
                valid_end -= 1
            valid_entities.append([valid_start, valid_end, label])
        cleaned_data.append([text, {'entities': valid_entities}])
    return cleaned_data

def convert_dataturks_to_spacy(json_file_path):
    try :
        training_data = []
        lines=[]
        with open(json_file_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]
                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1 ,label))
            training_data.append((text, {"entities" : entities}))
        return training_data
    except Exception :
        logging.exception("Unable to process " + json_file_path)
        return None

TRAIN_DATA = entity_span_trim(convert_dataturks_to_spacy("/Users/sagar_19/Desktop/BRT_Approach2/src/traindata.json"))

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(
    model=None,
    new_model_name="training",
    output_dir='/Users/sagar_19/Desktop/testing/model',
    n_iter=30
):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy

    if "ner" not in nlp.pipe_names:
        print("Creating new pipe")
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)

    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "Marathwada Mitra Mandals College of Engineering"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)
