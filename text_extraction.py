from pdfminer.converter import TextConverter
# interpreter to process the page content
from pdfminer.pdfinterp import PDFPageInterpreter
# used to store shared resources such as fonts and images
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
import PyPDF2
from io import StringIO
import docx2txt
import subprocess

#extract text from pdf
def extract_text_from_pdf(pdf_path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    fp = open(pdf_path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    raw_resume_text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    return raw_resume_text

# extract text from docx
def extract_text_from_docx(docx_path) :
    raw_resume_text = docx2txt.process(docx_path)
    if raw_resume_text :
        raw_resume_text = [line.replace('\t', ' ') for line in raw_resume_text.split('\n') if line]
        return ' '.join(raw_resume_text)
    return None

# extract text from doc
def extract_text_from_doc(doc_path) :
    try :
        # execute a child program in a new process
        process = subprocess.Popen(['catdoc', '-w', doc_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    except(FileNotFoundError, ValueError, subprocess.TimeoutExpired, subprocess.SubprocessError) as err:
        return(None, str(err))
    else :
        '''Interact with process: Send data to stdin. Read data from stdout and stderr, 
        until end-of-file is reached. Wait for process to terminate and set the returncode 
        attribute. The optional input argument should be data to be sent to the child process, 
        or None, if no data should be sent to the child. If streams were opened in text mode, 
        input must be a string. Otherwise, it must be bytes.'''
        stdout = process.communicate()
        # remove leading and trailing spaces
        return stdout[0].strip()


# getting extension of resume
def get_resume_extension(file_path) :
    # find out substring after . character
    extension = file_path.partition(".")[2]
    return extension

# wrapper function for extractig the text
def extract_text(file_path) :
    raw_resume_text = ''
    extension = get_resume_extension(file_path)
    if extension == "pdf" :
        raw_resume_text = extract_text_from_pdf(file_path)
    elif extension == "docx" :
        raw_resume_text = extract_text_from_docx(file_path)
    elif extension == "doc" :
        raw_resume_text = extract_text_from_doc(file_path)
    return raw_resume_text

# get number of pages in pdf
def get_pdf_no_of_pages(pdf_path) :
    file = open(pdf_path, 'rb')
    readpdf = PyPDF2.PdfFileReader(file)
    totalpages = readpdf.numPages
    return totalpages