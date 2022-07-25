# This will be used when user wanmt to use NLP approach for resume analysis but do not want to give input skills

import pandas as pd
import numpy as np

# spacy
import spacy
nlp = spacy.load("nl_core_news_sm")
from spacy import displacy

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import hstack
from sklearn import metrics
import pickle

def read_resume_dataset(csv_path) :
    df = pd.read_csv(csv_path)
    df = df.reindex(np.random.permutation(df.index))
    data = df.copy().iloc[0:500]
    return data, df

'''Entity ruler helps us add additional rules to highlight various categories 
    within the text, such as skills and job description in our case.'''
def pipeline_newruler_adder(skill_patterns_path) :
    new_ruler = nlp.add_pipe("entity_ruler")
    new_ruler.from_disk(skill_patterns_path)
    return new_ruler

def skills_extract(resume_text) :
    # type of doc - <class 'spacy.tokens.doc.Doc'>
    doc = nlp(resume_text)
    my_set = []
    sub_set = []
    for ent in doc.ents :
        #print(ent.label_)
        #print(type(ent.label_))
        # tokens format -> SKILL|Python
        if ent.label_[0:5] == "SKILL" :
            sub_set.append(ent.text)
            #print(sub_set)
    my_set.append(sub_set)
    return sub_set

def unique_skills(skills) :
    #print(type(skills))
    return list(set(skills))

# cleaning of the text using nltk
def clean_resume_text(data) :
    clean_text = []
    for i in range(data.shape[0]) :
        # regex to remove hyperlinks, special characters, or punctuations.
        resume_text = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"', " ", data["Resume_str"].iloc[i])
        # Lowering text
        resume_text = resume_text.lower()
        # Splitting text into array based on space
        resume_text = resume_text.split()
        # Lemmatizing text to its base form for normalizations
        lm = WordNetLemmatizer()
        # removing English stopwords
        resume_text = [lm.lemmatize(word) for word in resume_text if not word in set(stopwords.words("english"))]
        resume_text = " ".join(resume_text)
        # Appending the results into an array.
        clean_text.append(resume_text)
        #print(clean_text)
    return clean_text

#arg 1: data : DataFrame 
#arg 2: clean_text : List  
def modify_resume_csv(data, clean_text) :
    data["clean_resume"] = clean_text
    #print(type(data["clean_resume"].str))
    '''apply() method. This function acts as a map() function in Python. It takes a function as an 
       input and applies this function to an entire DataFrame.'''
    data["skills"] = data["clean_resume"].str.lower().apply(skills_extract)
    data["skills"] = data["skills"].apply(unique_skills)
    return data

def custom_NER(data, df, new_ruler) :
    patterns = df.Category.unique()
    for a in patterns :
        new_ruler.add_patterns([{"label" : "Job-Category", "pattern" : a}])
    # options=[{"ents": "Job-Category", "colors": "#ff3232"},{"ents": "SKILL", "colors": "#56c426"}]
    colors = {
        "Job-Category": "linear-gradient(90deg, #aa9cfc, #fc9ce7)",
        "SKILL": "linear-gradient(90deg, #9BE15D, #00E3AE)",
        "ORG": "#ffd966",
        "PERSON": "#e06666",
        "GPE": "#9fc5e8",
        "DATE": "#c27ba0",
        "ORDINAL": "#674ea7",
        "PRODUCT": "#f9cb9c",
    }
    options = {
        "ents": [
            "Job-Category",
            "SKILL",
            "ORG",
            "PERSON",
            "GPE",
            "DATE",
            "ORDINAL",
            "PRODUCT",
        ],
        "colors": colors,
    }
    sent = nlp(data["Resume_str"].iloc[5])
    return displacy.render(sent, style="ent", options=options)

'''LabelEncoder can be used to normalize labels. 
   It can also be used to transform non-numerical labels (as long as they are hashable and comparable) to numerical labels.'''
'''we will encode the ‘Category’ column using LabelEncoding. Even though the ‘Category’ column is ‘Nominal’ data we are using 
   LabelEncong because the ‘Category’ column is our ‘target’ column. By performing LabelEncoding each category will become a class 
   and we will be building a multiclass classification model'''
def label_encoding(data) :
    label_encoder = LabelEncoder()
    data["Category"] = label_encoder.fit_transform(data["Category"])
    return data

'''Vectorization or word embedding is the process of converting text data to numerical vectors'''
def word_vectorizer(data) :
    required_text = data["clean_resume"].values
    #print(type(required_text))
    required_target = data["Category"].values
    # Transforms text to feature vectors that can be used as input to estimator
    # Sublinear tf-scaling is modification of term frequency, which calculates weight
    wordVectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
    '''maps each single token to a position in the output matrix. Fitting on the training set and transforming on the training and test set assures that, given a word, 
    the word is correctly always mapped on the same column, both in the training and test set. '''
    wordVectorizer.fit(required_text)
    #transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text
    wordFeatures = wordVectorizer.transform(required_text)
    # print("Feature completed ")
    # divide in train dataset and test dataset
    X_train,X_test,y_train,y_test = train_test_split(wordFeatures,required_target,random_state=0, test_size=0.2, shuffle=True, stratify=required_target)
    #print(type(X_test))
    return X_train, X_test, y_train, y_test

def predict_classifier(X_train, X_test, y_train, y_test) :
    # when we want to do multiclass or multilabel classification and it's strategy consists of fitting one classifier per class. For each classifier, the class is fitted against all the other classes.
    # The K in the name of this classifier represents the k nearest neighbors, where k is an integer value specified by the user. Generally it is 5.
    # train the model
    classifier = OneVsRestClassifier(KNeighborsClassifier()).fit(X_train, y_train)
    # prediction for each test instance
    predict = classifier.predict(X_test)
    #pickle.dump(classifier, open('model.pkl','wb'))
    #model = pickle.load(open('model.pkl','rb'))
    #required_text = df["clean_resume"].values
    #wordVectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english')
    #wordVectorizer.fit(required_text)
    #wordFeatures = wordVectorizer.transform(required_text)
    #print(type(wordFeatures))
    #print(classifier.predict(wordFeatures))
    training_accuracy = classifier.score(X_train, y_train)
    testing_accuracy = classifier.score(X_test, y_test)
    # precision    recall  f1-score   support
    report_classifier = metrics.classification_report(y_test, predict)
    return training_accuracy, testing_accuracy, report_classifier

if __name__ == "__main__" :
    print("\nPlease wait till your dataset is being processed for training classifier......")
    skill_patterns_path = "/Users/sagar_19/Desktop/BRT/dataset/skills_pattern.jsonl"
    csv_path = "/Users/sagar_19/Desktop/BRT/dataset/resume_dataset.csv"
    print("\nReading the dataset and converting into dataframe ....")
    data, df1 = read_resume_dataset(csv_path)
    print("\nAdding new ruler in NER pipeline for skills recognition from resumes...")
    new_ruler = pipeline_newruler_adder(skill_patterns_path)
    print("\nCleaning resumes in the dataset for further processing....")
    clean_text = clean_resume_text(data)
    print("\nModifying the dataset....")
    data = modify_resume_csv(data, clean_text)
    print("\nHighlighting the entities in 5 resumes from given dataset....")
    markup_text = custom_NER(data, df1, new_ruler)
    f2 = open("output.html", "w")
    for line in markup_text :
        f2.write(line)
    f2.close()
    print("\nYou can see highlighted entities of first resume of dataset - just open output.html in browser")
    print("\nEncoding labels....")
    data = label_encoding(data)
    print("\nApplying word vectorizer....")
    try :
        X_train, X_test, y_train, y_test = word_vectorizer(data)
        print("\nDoing prediction by classifier....\n")
        training_accuracy, testing_accuracy, report_classifier = predict_classifier(X_train, X_test, y_train, y_test)
        #pickle.dump(classifier, open('model.pkl','wb'))
        #model = pickle.load(open('model.pkl','rb'))
        print('\nAccuracy of KNeighbors Classifier on training set: {:.2f}'.format(training_accuracy))
        print('\nAccuracy of KNeighbors Classifier on test set: {:.2f}'.format(testing_accuracy))
        print("\n Classification report for classifier - KNeighborsClassifier :\n%s\n" % (report_classifier))
    except ValueError :
        print("\nIn given dataset, y class has only 1 member whichh is too few for prediction by classifier")