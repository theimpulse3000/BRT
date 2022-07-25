import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import text_extraction as te

# cleaning of the text using nltk
def extract_clean_text(raw_resume_text) :
    clean_text = []
    # regex to remove hyperlinks, special characters, or punctuations.
    resume_text = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?"', " ", raw_resume_text)
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
    return clean_text

if __name__ == "__main__" :
    doc_path = "/Users/sagar_19/Desktop/BRT/resumes/Swapnil_resume.doc"
    raw_resume_text = te.extract_text(doc_path)
    clean_text = extract_clean_text(raw_resume_text)
    #print(clean_text)
    clean_text = ' '.join(clean_text) #convert that list into one string
    #print(clean_text)
    clean_text = clean_text.split() # convert that string into list
    print(clean_text)