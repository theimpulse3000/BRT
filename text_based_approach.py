import text_cleaning as tc
import text_extraction as te
from prettytable import PrettyTable
import os
import shutil

# this if user given the required skills
def resume_analysis_by_text(clean_text, required_skills) :
    clean_text = ' '.join(clean_text) #convert that list into one string
    #print(clean_text)
    clean_text = clean_text.split() # convert that string into list
    #but this time, this list will contain each word in the resume as element of the list
    #print(clean_text)
    #list[whole resume as one string] -> one string -> list[each word in resume as element]
    required_skills = required_skills.lower().split(",") # returns list
    present_skillset = []
    for skill in required_skills :
        for word in clean_text :
            if skill == word :
                if skill in present_skillset :
                    continue
                present_skillset.append(skill)
    #print(present_skillset)
    matching_score = (len(present_skillset) / len(required_skills)) * 100
    return matching_score

# if user doesn't give the required skills then use information already present in the system
# for this dictionary containing job description and keywords as key value pair should be present
terms = {'Quality/Six Sigma':['black belt','capability analysis','control charts','doe','dmaic','fishbone',
                              'gage r&r', 'green belt','ishikawa','iso','kaizen','kpi','lean','metrics',
                              'pdsa','performance improvement','process improvement','quality',
                              'quality circles','quality tools','root cause','six sigma',
                              'stability analysis','statistical analysis','tqm'],      
        'Operations management':['automation','bottleneck','constraints','cycle time','efficiency','fmea',
                                 'machinery','maintenance','manufacture','line balancing','oee','operations',
                                 'operations research','optimization','overall equipment effectiveness',
                                 'pfmea','process','process mapping','production','resources','safety',
                                 'stoppage','value stream mapping','utilization'],
        'Supply chain':['abc analysis','apics','customer','customs','delivery','distribution','eoq','epq',
                        'fleet','forecast','inventory','logistic','materials','outsourcing','procurement',
                        'reorder point','rout','safety stock','scheduling','shipping','stock','suppliers',
                        'third party logistics','transport','transportation','traffic','supply chain',
                        'vendor','warehouse','wip','work in progress'],
        'Project management':['administration','agile','budget','cost','direction','feasibility analysis',
                              'finance','kanban','leader','leadership','management','milestones','planning',
                              'pmi','pmp','problem','project','risk','schedule','scrum','stakeholders'],
        'Data analytics':['analytics','api','aws','big data','busines intelligence','clustering','code',
                          'coding','data','database','data mining','data science','deep learning','hadoop',
                          'hypothesis test','iot','internet','machine learning','modeling','nosql','nlp',
                          'predictive','programming','python','r','sql','tableau','text mining',
                          'visualuzation']}

def get_count_list(terms) :
  count_dic = {k:len(v) for k,v in terms.items()}
  count =list(count_dic.values())
  #print(count)
  return count

def resume_analysis(text, count) :
  # Initialize counter for each area 
  quality = operations = supplychain = project = data = 0
  scores = []
  for area in terms.keys():
    if area == "Quality/Six Sigma" :
      for word in terms[area]:
        if word in text:
          quality = quality + 1
      quality = (quality / count[0]) * 100
      scores.append(quality)
  
    elif area == 'Operations management':
          for word in terms[area]:
              if word in text:
                  operations +=1
          operations = (operations / count[1]) * 100
          scores.append(operations)
        
    elif area == 'Supply chain':
          for word in terms[area]:
              if word in text:
                  supplychain +=1
          supplychain = (supplychain / count[2]) * 100
          scores.append(supplychain)
        
    elif area == 'Project management':
          for word in terms[area]:
              if word in text:
                  project +=1
          project = (project / count[3]) * 100
          scores.append(project)
        
    elif area == 'Data analytics':
          for word in terms[area]:
              if word in text:
                  data +=1
          data = (data / count[4]) * 100
          scores.append(data)
  return scores

def is_shortlist(scores, path, threshold_percentage) :
  names = ['Quality/Six Sigma', 'Operations management', 'Supply chain', 'Project management','Data analytics']
  '''t = PrettyTable(['Job Profile', 'Scores'])
  j = 0
  for j,item in enumerate(names, start=j):
      t.add_row([names[j], scores[j]])
  print(t)'''
  flag = 0
  for i in range(len(scores)) :
    if scores[i] >= threshold_percentage :
        print("\n%s is shortlisted for %s with %d percentage matching\n" %(path, names[i], int(scores[i])))
        flag = 1
  if flag == 0:
    print("\n%s is not shortlisted for any job description available in the system" %path)
    
if __name__ == "__main__" :
    threshold_percentage = int(input("\nEnter threshold for shortlisting : "))
    choice = int(input("\nEnter the choice : 1)Based on required skills given by you 2)Based on database already present in the system :  "))
    print("\n")
    src_dir = "/Users/sagar_19/Desktop/BRT/resumes/"
    dest_dir = "/Users/sagar_19/Desktop/BRT/shortlisted/"

    if(choice == 1):
        required_skills = input("\nEnter required skills here : ")
        resumes = []
        data = []
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
            matching_score = resume_analysis_by_text(clean_text, required_skills)
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
    
    if(choice == 2):
        resumes = []
        data = []
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
            count = get_count_list(terms)
            scores = resume_analysis(clean_text, count)
            path = os.path.basename(os.path.normpath(resumes[i]))
            is_shortlist(scores, path, threshold_percentage)
