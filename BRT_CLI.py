import argparse
import os
import nlp_with_dataset as nd
import text_based_approach as tb
from text_based_approach import terms
import resume_parser as rp
import shutil
import multiprocessing as multp
import pprint
import json

import text_extraction as te
import text_cleaning as tc
import nlp_with_input as ni

parser = argparse.ArgumentParser()
parser.add_argument(
    '-f',
    '--skills_file',
    type=str,
    #required=True,
    help = "Text file containing skills you want to look for"
)
parser.add_argument(
    '-th',
    '--threshold_percentage',
    type=int,
    #required=True,
    help = "Threshold you want to set for shortlisting"
)
parser.add_argument(
    '-i',
    '--input_folder',
    type=str,
    #required=True,
    help = "Name of folder which contains all resumes for processing"
)
parser.add_argument(
    '-o',
    '--output_folder',
    type=str,
    #required=True,
    help = "Name of folder which contains shortlisted resumes"
)
parser.add_argument(
    '-c',
    '--choice',
    type=str,
    #required=True,
    help="* Text-Based Approach - 1.1)Using given required skills 1.2)Using information present in the system  * NLP Approach - 2.1)Using given required skills 2.2)Using information present in the system"
)
# for datasets
parser.add_argument(
    '-sp',
    '--skills_pattern',
    type=str,
    help="path for skills pattern json file which will be used by NER"
)
# for datasets
parser.add_argument(
    '-ds',
    '--resume_csv_dataset',
    type=str,
    help="path for resume dataset "
)

def banner():
    banner_string = r'''
             _ _ _ _     _ _ _ _    _ _ _ _ 
            /      /    /      /       /
           /______/    /_ _ _ /       /
          /      /    /     \        /
         /__ _ _/    /       \      /

         Barclays Recruitment Tool
    '''
    print(banner_string)

args = parser.parse_args()
    # args[0] = input skills
    # args[1] = threshold percentage
    # args[2] = input folder
    # args[3] = output_folder
    # args[4] = choice
    # args[5] = skill patterns
    # args[6] = resume csv file

def main() :
    banner()
    cwd = os.path.dirname(os.path.abspath(__file__))
    cwd = cwd + "/"
    input_skills_file_path = cwd + args.skills_file
    with open(input_skills_file_path, "r") as f:
        input_skills_list = f.readlines()
    input_skills = ""
    for i in input_skills_list :
        input_skills = input_skills + "," + i
    input_skills = input_skills[1:]
    threshold_percentage = int(args.threshold_percentage)
    src_dir = cwd + args.input_folder + "/"
    dest_dir = cwd + args.output_folder + "/"
    skill_patterns_path = cwd + "dataset/" + args.skills_pattern
    csv_path = cwd + "dataset/" + args.resume_csv_dataset
    #print(input_skills_file_path, threshold_percentage, src_dir, dest_dir, skill_patterns_path, csv_path)
    if args.choice == '1.1' :
        resumes = []
        print("\nCollecting all resumes from source folder......")
        for dirpath, dirnames, filenames in os.walk(src_dir) :
            for resume_file in filenames :
                file = os.path.join(dirpath, resume_file)
                resumes.append(file)
        start = 1
        end = len(resumes)
        #print(resumes)
        print("\nDeleting the resumes if pre-exist in shortlisted folder.....")
        for file_name in os.listdir(dest_dir):
            #print("In delete function")
            # construct full file path
            file = dest_dir + file_name
            #print(file)
            if os.path.isfile(file):
                #print('Deleting file:', file)
                os.remove(file)
        print("\nAnalysing each resume in resume folder for shortlisting......")
        for i in range(start, end) :
            resume_path = resumes[i]
            raw_resume_text = te.extract_text(resume_path)
            #following function returns list containing whole resume text as one string. So list contains only one element and that is one string
            clean_text = tc.extract_clean_text(raw_resume_text)
            matching_score = tb.resume_analysis_by_text(clean_text, input_skills)
            if matching_score >= threshold_percentage :
                basename = os.path.basename(resumes[i])
                print("\n%s is shortlisted." %basename)
                src_path = resume_path
                shutil.copy2(src_path, dest_dir)
        shortlisted_resumes = []
        print("\nMoving shortlisted resumes into shortlisted folder...")
        for dirpath, dirnames, filenames in os.walk(dest_dir) :
            #print("In result list")
            for resume_file in filenames :
                file = os.path.join(dirpath, resume_file)
                shortlisted_resumes.append(file)
    if args.choice == "1.2" :
        resumes = []
        for dirpath, dirnames, filenames in os.walk(src_dir) :
            for resume_file in filenames :
                file = os.path.join(dirpath, resume_file)
                resumes.append(file)
        start = 1
        end = len(resumes)
        #print(resumes)
        for i in range(start, end) :
            resume_path = resumes[i]
            resume_text = te.extract_text(resume_path)
            clean_text = tc.extract_clean_text(resume_text)
            clean_text = ' '.join(clean_text) #convert that list into one string
            #print(clean_text)
            clean_text = clean_text.split() # convert that string into list
            count = tb.get_count_list(terms)
            scores = tb.resume_analysis(clean_text, count)
            path = os.path.basename(os.path.normpath(resumes[i]))
            tb.is_shortlist(scores, path, threshold_percentage)
    if args.choice == "2.1" :
        print("\nPlease wait till your dataset is being processed for training classifier......")
        print("\nReading the dataset and converting into dataframe ....")
        data, df1 = nd.read_resume_dataset(csv_path)
        print("\nAdding new ruler in NER pipeline for skills recognition from resumes...")
        new_ruler = nd.pipeline_newruler_adder(skill_patterns_path)
        print("\nCleaning resumes in the dataset for further processing....")
        clean_text = nd.clean_resume_text(data)
        print("\nModifying the dataset....")
        data = nd.modify_resume_csv(data, clean_text)
        print("\nHighlighting the entities in 5 resumes from given dataset....")
        markup_text = nd.custom_NER(data, df1, new_ruler)
        f2 = open("output.html", "w")
        for line in markup_text :
            f2.write(line)
        f2.close()
        print("\nYou can see highlighted entities of first resume of dataset - just open output.html in browser")
        print("\nEncoding labels....")
        data = nd.label_encoding(data)
        print("\nApplying word vectorizer....")
        try :
            X_train, X_test, y_train, y_test = nd.word_vectorizer(data)
            print("\nDoing prediction by classifier....\n")
            training_accuracy, testing_accuracy, report_classifier = nd.predict_classifier(X_train, X_test, y_train, y_test)
            #pickle.dump(classifier, open('model.pkl','wb'))
            #model = pickle.load(open('model.pkl','rb'))
            print('\nAccuracy of KNeighbors Classifier on training set: {:.2f}'.format(training_accuracy))
            print('\nAccuracy of KNeighbors Classifier on test set: {:.2f}'.format(testing_accuracy))
            print("\n Classification report for classifier - KNeighborsClassifier :\n%s\n" % (report_classifier))
        except ValueError :
            print("\nIn given dataset, y class has only 1 member whichh is too few for prediction by classifier")
    if args.choice == "2.2" :
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
            match = ni.matching_score(input_skills, clean_text_str)
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