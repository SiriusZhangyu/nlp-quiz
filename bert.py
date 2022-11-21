import PyPDF2
import textract
import nltk
from autocorrect import Speller
import re
from nltk.corpus import stopwords, brown
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import spacy
from spacy import displacy

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


filename = './2022.pdf'
open_filename = open(filename, 'rb')
ind_manifesto = PyPDF2.PdfFileReader(open_filename)
ind_manifesto.getDocumentInfo()
total_pages = ind_manifesto.numPages
count = 0
text = ''


while (count < total_pages):
    mani_page = ind_manifesto.getPage(count)
    count += 1
    text += mani_page.extractText()
if text != '':
    text = text
else:
    textract.process(open_filename, method='tesseract', encoding='utf-8', langauge='eng')


def to_lower(text):

    spell = Speller(lang='en')
    texts = spell(text)
    return ' '.join([w.lower() for w in word_tokenize(text)])

lower_case = to_lower(text)


def clean_text(lower_case):
    words = nltk.word_tokenize(lower_case)
    # punctuations = ['.', ',', '/', '!', '?', ';', ':', '(', ')', '[', ']', '-', '_', '%']
    punctuations = re.sub(r'\W', ' ', str(lower_case))
    stop_words = stopwords.words('english')
    w_num = re.sub('\w*\d\w*', '', lower_case).strip()
    lower_case = re.sub(r'\s+[a-zA-Z]\s+', ' ', lower_case)
    lower_case = re.sub(r'\s+', ' ', lower_case, flags=re.I)
    lower_case = re.sub(r'^b\s+', '', lower_case)
    lower_case = re.sub(r'^b\s+', '', lower_case)
    keywords = [word for word in words if not word in stop_words and word in punctuations and word in w_num]
    return keywords

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in clean_text(lower_case)]
clean_data = ' '.join(lemmatized_word)


df = pd.DataFrame([clean_data])
df.columns = ['script']
df.index = ['Itula']


# ENTITY EXTRACTION

def print_entities(pipeline, text):
    # Create a document
    document = pipeline(text)

    # Entity text & label extraction
    for entity in document.ents:
        print(entity.text + '->', entity.label_)

def visualize_entities(pipeline, text):
    # Create a document
    document = pipeline(text)

    # Show entities in pretty manner
    displacy.render(document, jupyter=True, style='ent')


nlp_sm = spacy.load("en_core_web_lg")
print_entities(nlp_sm, clean_data)






# SUMMARIZE


from transformers import DistilBertModel, DistilBertTokenizer
from summarizer import Summarizer,TransformerSummarizer


bert_model = Summarizer()
bert_summary = ''.join(bert_model(lower_case, min_length=60))
print(bert_summary)


#TOPIC MODELING
from bertopic import BERTopic


model = BERTopic(language='english')
topics, probabilities = model.fit_transform(list(lower_case.split('.')))

print(model.get_topic_info())