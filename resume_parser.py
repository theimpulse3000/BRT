import os
import spacy
from spacy.matcher import Matcher
import text_extraction as te
import extracting_information as ei

class Resume_Parser(object) :
    def __init__(self, resume, custom_regex = None) :
        nlp = spacy.load("nl_core_news_sm")
        custom_nlp = spacy.load(os.path.dirname(os.path.abspath(__file__)))
        self.__matcher = Matcher(nlp.vocab)
        self.__details = {
            'name' : None,
            'mobile_number' : None,
            'email' : None,
            'college_name' : None,
            'degree' : None,
            'designation' : None,
            'experience' : None,
            'company_names' : None,
            #'pdf_no_of_pages' : None,
            'total_experience' : None
        }
        self.__resume = resume
        self.__raw_text = te.extract_text(self.__resume)
        # with cleaning of text
        '''self.__clean_raw_text = functionalities.clean_my_resume_text(self.__raw_text)
        self.__resume_text = ' '.join(self.__clean_raw_text)'''
        # without cleaning of text
        self.__resume_text = ' '.join(self.__raw_text.split()) 
        self.__nlp = nlp(self.__resume_text)
        self.__custom_nlp = custom_nlp(self.__raw_text)
        #self.__noun_chunks = list(self.__nlp.noun_chunks)
        self.__get_basic_details()
        #print(self.__details)
    
    def get_extracted_details(self) :
        #print(self.__details)
        return self.__details

    def __get_basic_details(self) :
        cust_entities = ei.extract_entities_with_NER(self.__custom_nlp)
        name = ei.extract_name(self.__nlp, matcher = self.__matcher)
        mobile_number = ei.extract_mobile_number(self.__resume_text)
        email = ei.extract_email_address(self.__resume_text)
        resume_entities = ei.grad_entity_sections_extract(self.__raw_text)
        try :
            self.__details['name'] = cust_entities['Name'][0]
        except (IndexError, KeyError) :
            self.__details['name'] = name
        self.__details['email'] = email
        self.__details['mobile_number'] = mobile_number
        try :
            self.__details['college_name'] = resume_entities['College Name']
        except KeyError :
            pass
        try :
            self.__details['degree'] = cust_entities['Degree']
        except KeyError :
            pass
        try :
            self.__details['designation'] = cust_entities['Designation']
        except KeyError :
            pass
        try :
            self.__details['company_names'] = cust_entities['Companies worked at']
        except KeyError :
            pass
        try :
            self.__details['experience'] = resume_entities['experience']
            try :
                expr = round(ei.total_experience(resume_entities['experience']))
                self.__details['total_experience'] = expr
            except KeyError :
                self.__details['total_experience'] = 0
        except KeyError :
            self.__details['total_experience'] = 0
        #print(self.__details)
        return

def resume_result_wrapper(resume) :
    parser = Resume_Parser(resume)
    return parser.get_extracted_details()