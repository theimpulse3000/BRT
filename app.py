import dataset_BRT_approach as dsa
import functionalities as fn
import multiprocessing as multp
import resume_parser as rp
import shutil
import json
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from flask_script import Manager, Server
import sys

app=Flask(__name__)
manager = Manager(app)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['pdf', 'docx', 'doc'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        skills_text = request.form.get("skills_text")
        with open("skills.txt", "w") as f :
            f.write(skills_text)
        f.close()

        flash('File(s) successfully uploaded and your required skills are saved successfully')
    
        #return render_template("result_show.html")
        return('/')

def main() :
        print("Hi", file = sys.stdout)
        skill_patterns_path = "/Users/sagar_19/Desktop/BRT_Approach2/src/dataset/skills_pattern.jsonl"
        threshold_percentage = request.form.get("threshold")
        new_ruler = dsa.pipeline_newruler_adder(skill_patterns_path)
        # About pool object
        '''Pool object which offers a convenient means of parallelizing the execution 
         of a function across multiple input values, distributing the input data across processes (data parallelism)'''
        # About cpu_count()
        '''One of the useful functions in multiprocessing is cpu_count() . This returns the number 
        of CPUs (computer cores) available on your computer to be used for a parallel program'''
        pool = multp.Pool(multp.cpu_count())
        resumes = []
        data = []
        src_dir = "/Users/sagar_19/Desktop/BRT_Approach2/uploads/"
        for dirpath, dirnames, filenames in os.walk(src_dir) :
            for resume_file in filenames :
                file = os.path.join(dirpath, resume_file)
                resumes.append(file)
        dest_dir = "/Users/sagar_19/Desktop/BRT_Approach2/shortlisted/"
        print("\nDeleting the files if pre-exist in shortlisted folder.....", file = sys.stdout)
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
            resume_text = fn.extract_text(resume_path)
            markup_resume_text = dsa.use_entity_recg_for_resume(resume_text)
            #print(markup_resume_text)
            #print(resume_text)
            clean_text = fn.clean_my_resume_text(resume_text)            
            #print(type(clean_text))
            clean_text_str = ""
            for i in clean_text:
                clean_text_str += i
            #print(clean_text_str)
            f = open("skills.txt", "r")
            input_skills = f.read()
            match = dsa.matching_score(input_skills, clean_text_str)
            #print(match)
            if(match >= threshold_percentage) :
                src_path = resume_path
                shutil.copy2(src_path, dest_dir)
                #print('Copied')
        shortlisted_resumes = []
        print("\nMoving shortlisted resume into shortlisted folder...", file = sys.stdout)
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
        #print("\nThe details of shortlisted candidates are as follows : \n")
        #pprint.pprint(results)
        #print("\n\n")
        with open('result.json', 'w') as f:
            json.dump(results, f)

@manager.command
def runserver() :
    app.run(host='127.0.0.1',port=5000,debug=False,threaded=True)
    main()

if __name__ == "__main__":
        manager.run()