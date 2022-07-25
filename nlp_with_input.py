from nltk import tokenize 
from operator import itemgetter
import math
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# spacy
import spacy
nlp = spacy.load("nl_core_news_sm")
from spacy import displacy 

import nlp_with_dataset as nd
import multiprocessing as multp
import os
import shutil
import pprint

import text_extraction as te
import text_cleaning as tc
import resume_parser as rp

# helper function for most_used
def check_sent(word, sentences): 
    final = [all([w in x for w in word]) for x in sentences] 
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

# helper function for most_used
def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
    return result

def most_used(resume_text) :
    stop_words = set(stopwords.words('english'))
    doc = resume_text
    total_words = doc.split()
    total_word_length = len(total_words)
    #print(total_word_length)
    total_sentences = tokenize.sent_tokenize(doc)
    total_sent_len = len(total_sentences)
    #print(total_sent_len)
    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1
    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
    #print(tf_score)
    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1
    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())
    #print(idf_score)
    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
    #print(tf_idf_score)
    return get_top_n(tf_idf_score, 5)

def use_entity_recg_for_resume(resume_text) :
    resume_text = nlp(resume_text)
    # this will output in markup
    return displacy.render(resume_text, style = "ent")

def matching_score(input_skills, resume_text) : 
    #print('~~~~~~~~~~~~Starting def matching_score')
    req_skills = input_skills.lower().split(",")
    #print(req_skills)
    #print(resume_text.lower())
    resume_skills = nd.unique_skills(nd.skills_extract(resume_text.lower()))
    #print(resume_skills)
    score = 0
    for i in req_skills:
        if i in resume_skills:
            score = score + 1
    req_skills_len = len(req_skills)
    match = round(score / req_skills_len * 100, 1)
    return match

def main() :
    input_skills = input("\nEnter here skills required : ")
    threshold_percentage = int(input("\nEnter threshold you want to set for shortlisting : "))
    skill_patterns_path = "/Users/sagar_19/Desktop/BRT/dataset/skills_pattern.jsonl"
    main_dir = "/Users/sagar_19/Desktop/BRT/"
    src_dir = input("\nEnter input folder name : ")
    src_dir = main_dir + src_dir + "/"
    dest_dir = input("\nEnter output output folder name : ")
    dest_dir = main_dir + dest_dir + "/"
    print("\nAdding new ruler in NER pipeline for skills recognition from resumes...")
    new_ruler = nd.pipeline_newruler_adder(skill_patterns_path)
    # About pool object
    '''Pool object which offers a convenient means of parallelizing the execution 
    of a function across multiple input values, distributing the input data across processes (data parallelism)'''
    # About cpu_count()
    '''One of the useful functions in multiprocessing is cpu_count() . This returns the number 
    of CPUs (computer cores) available on your computer to be used for a parallel program'''
    pool = multp.Pool(multp.cpu_count())
    resumes = []
    data = []
    for dirpath, dirnames, filenames in os.walk(src_dir) :
        for resume_file in filenames :
            file = os.path.join(dirpath, resume_file)
            resumes.append(file)
    print("\nDeleting the resumes if already exist in shortlisted folder.....")
    for file_name in os.listdir(dest_dir):
        #print("In delete function")
        # construct full file path
        file = dest_dir + file_name
        #print(file)
        if os.path.isfile(file):
            #print('Deleting file:', file)
            os.remove(file)
    start = 1
    end = len(resumes) + 1
    for resume_path in resumes[start : end] :
        resume_text = te.extract_text(resume_path)
        #print(resume_text)
        #markup_resume_text = use_entity_recg_for_resume(resume_text)
        #print(markup_resume_text)
        clean_text = tc.extract_clean_text(resume_text)            
        #print(type(clean_text))
        clean_text_str = ""
        for i in clean_text:
            clean_text_str += i
        #print(clean_text_str)
        match = matching_score(input_skills, clean_text_str)
        #print(match)
        if(match >= threshold_percentage) :
            src_path = resume_path
            shutil.copy2(src_path, dest_dir)
            #print('Copied')
    print("\nMoving shortlisted resume into shortlisted folder...")
    shortlisted_resumes = []
    for dirpath, dirnames, filenames in os.walk(dest_dir) :
        #print("In result list")
        for resume_file in filenames :
            file = os.path.join(dirpath, resume_file)
            shortlisted_resumes.append(file)
    #print(shortlisted_resumes)
    #start = 1
    #end = len(shortlisted_resumes)
    results = [pool.apply_async(rp.resume_result_wrapper, args = (i,)) for i in shortlisted_resumes]
    #results = results.append(results)
    #print(results)
    results = [p.get() for p in results]
    #print(results)
    print("\nThe details of shortlisted candidates are as follows : \n")
    pprint.pprint(results)
    print("\n\n")

if __name__ == "__main__" :
    main()