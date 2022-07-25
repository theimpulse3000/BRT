import datetime
import entity_patterns as ep
from dateutil import relativedelta
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


# Helper function to extract different entities with custom trained model using SpaCy's NER
def extract_entities_with_NER(nlp_resume_text) :
    entities = {}
    for ent in nlp_resume_text.ents :
        if ent.label_ not in entities.keys() :
            entities[ent.label_] = [ent.text]
        else :
            entities[ent.label_].append(ent.text)
    for key in entities.keys() :
        entities[key] = list(set(entities[key]))
    return entities

# Helper function to extract all the raw text from sections of resume specifically for graduates and undergraduates
def grad_entity_sections_extract(resume_text) :
    key = False
    split_text = [i.strip() for i in resume_text.split('\n')]
    entities = {}
    for word_phrase in split_text :
        if len(word_phrase) == 1 :
            key_item = word_phrase
        else :
            key_item = set(word_phrase.lower().split()) & set(ep.GRAD_RESUME_SECTIONS)
        try :
            key_item = list(key_item)[0]
        except IndexError :
            pass
        if key_item in ep.GRAD_RESUME_SECTIONS :
            entities[key_item] = []
            key = key_item
        elif key and word_phrase.strip():
            entities[key].append(word_phrase)
    return entities

# Helper function to extract all the raw text from sections of resume specifically for professionals
def prof_entity_sections_extract(resume_text) :
    key = False
    split_text = [i.strip() for i in resume_text.split('\n')]
    entities = {}
    for word_phrase in split_text :
        if len(word_phrase) == 1 :
            key_item = word_phrase
        else :
            key_item = set(word_phrase.lower().split()) & set(ep.PROF_RESUME_SECTIONS)
        try :
            key_item = list(key_item)[0]
        except IndexError :
            pass
        if key_item in ep.PROF_RESUME_SECTIONS :
            entities[key_item] = []
            key = key_item
        elif key and word_phrase.strip():
            entities[key].append(word_phrase)
    return entities

# date1 : start date
# date2 : end date
def get_no_of_months(date1, date2) :
    if date2.lower() == "present" :
        date2 = datetime.now().strftime('%b %Y')
    try :
        if len(date1.split()[0]) > 3:
            date1 = date1.split()
            date1 = date1[0][:3] + ' ' + date1[1]
        if len(date2.split()[0]) > 3 :
            date2 = date2.split()
            date2 = date2[0][:3]+ ' ' + date2[1]
    except IndexError :
        return 0
    try :
        date1 = datetime.strptime(str(date1), '%b %Y')
        date2 = datetime.strptime(str(date2), '%b %Y')
        months_of_experience = relativedelta.relativedelta(date2, date1)
        months_of_experience = (months_of_experience.years * 12 + months_of_experience.months)
    except ValueError :
        return 0
    return months_of_experience 

# extract experience from resume text
# returns : list of experiences
def extract_experience(resume_text) :
    wordLemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    # tokenisation
    tokens_word = nltk.word_tokenize(resume_text)
    # remove stop words and lemmatize
    req_sentences = [w for w in tokens_word
        if w not in stopWords
        and wordLemmatizer.lemmatiza(w)
        not in stopWords]
    # a process to mark up the words in text format for a particular part of a speech based on its definition and context.
    sent = nltk.pos_tag(req_sentences)
    # parse regex
    ent_pattern = nltk.RegexParser('P: {<NNP>+}')
    cs = ent_pattern.parse(sent)
    test = []
    # for i in cs.subtrees(filter=lambda x: x.label() == 'P'):
    #     print(i)
    # VP = one of phrae tags
    for vp in list(cs.subtrees(filter = lambda x : x.label() == 'P')) :
            test.append(" ".join([i[0] for i in vp.leaves() if len(vp.leaves()) >= 2]))
    # search the wprd 'experience' in the text and then print the text after it
    req_text = [req_text[req_text.lower().index('experience') + 10 : ] for req_text in enumerate(test) if req_text and 'experience' in req_text.lower()]
    return req_text

# getting total months of experience
# list of experiences from above function
def total_experience(list_experience) :
    expr = []
    for line in list_experience :
        # re.I -> Perform case-insensitive matching; expressions like [A-Z] will also match lowercase letters
        exp = re.search(r'(?P<fmonth>\w+.\d+)\s*(\D|to)\s*(?P<smonth>\w+.\d+|present)', line, re.I)
        if exp :
            # groups() -> This method returns a tuple containing all the subgroups of the match, from 1 up to however many groups are in the pattern. 
            expr.append(exp.groups())
    total_exp = sum([get_no_of_months(i[0], i[2]) for i in expr])
    total_exp_months = total_exp
    return total_exp_months

# extract education
def extract_education(nlp_resume_text) :
    education = {}
    for index, text in enumerate(nlp_resume_text) :
        for text1 in text.split() :
            text1 = re.sub(r'[?|$|.|!|,]', r'', text1)
            if text1.upper() in ep.EDUCATION and text1 not in ep.STOP_WORDS :
                education[text1] = text + nlp_resume_text[index + 1]
    # now extract year
    edu_year = []
    for key in education.keys() :
        year = re.search(re.compile(ep.YEAR), education[key])
        if year :
            edu_year.append((key, ''.join(year.group(0))))
        else :
            edu_year.append(key)
    return edu_year

# extrating name from spacy nlp text 
# nlp_text: object of `spacy.tokens.doc.Doc
# matcher : object of spacy.matcher.Matcher
def extract_name(nlp_resume_text, matcher) :
    pattern = [ep.NAME_PATTERN]
    matcher.add("NAME", pattern)
    matches = matcher(nlp_resume_text)
    for _, start, end in matches :
        span = nlp_resume_text[start:end]
        return span.text

# extract mobile number
def extract_mobile_number(resume_text, custom_regex = None) :
    if not custom_regex :
        mobile_number_regex = r'''(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)
                              [-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'''
        phone_number = re.findall(re.compile(mobile_number_regex), resume_text)
    else :
        phone_number = re.findall(re.compile(custom_regex), resume_text)
    if phone_number :
        mobile_number = ''.join(phone_number[0])
        return mobile_number

# extract email address
def extract_email_address(resume_text) :
    email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", resume_text)
    if email :
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None

# exracting urls
def extract_url(text5):
    try:
     url = re.search("(?P<url>https?://[^\s]+)", text5).group("url")
    except:
        url = None
    return url

# extract address 
def extract_address(text):
    regexp = "[0-9]{1,3} .+, .+, [A-Z]{2} [0-9]{5}"
    address = re.findall(regexp, text)
    #addresses = pyap.parse(text, country='INDIA')
    return address

#find pincode
def extract_pincode(text):
    pincode =  r"[^\d][^a-zA-Z\d](\d{6})[^a-zA-Z\d]"
    pattern = re.compile(pincode)
    result = pattern.findall(text)
    if len(result)==0:
        return ' '
    return result[0]