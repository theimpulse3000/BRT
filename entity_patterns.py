from nltk.corpus import stopwords

NAME_PATTERN = [{'POS': 'PROPN'}, {'POS': 'PROPN'}] # proper noun

GRAD_RESUME_SECTIONS = [
    'accomplishments',
    'experience',
    'education',
    'interests',
    'skills',
    'certifications',
    'objective',
    'career objective',
    'summary',
    'leadership',
    'projects',
    'professional experience',
    'publications'
]

PROF_RESUME_SECTIONS = [
    'experience',
    'education',
    'interests',
    'professional experience',
    'publications',
    'skills',
    'certifications',
    'objective',
    'career objective',
    'summary',
    'leadership'
]

EDUCATION = [
            'BE', 'B.E.', 'B.E', 'BS', 'B.S', 'ME', 'M.E',
            'M.E.', 'MS', 'M.S', 'BTECH', 'MTECH',
            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII'
        ]

STOP_WORDS = set(stopwords.words('english'))

YEAR = r'(((20|19)(\d{2})))'