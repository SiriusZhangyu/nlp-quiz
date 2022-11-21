import PyPDF2
import textract
import nltk
from autocorrect import Speller
import re
from nltk.corpus import stopwords, brown
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from gensim.summarization.summarizer import summarize
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import spacy


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
    punctuations = ['.', ',', '/', '!', '?', ';', ':', '(', ')', '[', ']', '-', '_', '%']
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




# summarize


print(summarize(lower_case))


# topic analysis


data  = []
data.append(clean_text(lower_case))

def lemmatization(data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_output = []
    for sent in data:
        doc = nlp(" ".join(sent))
        texts_output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_output
nlp = spacy.load('en', disable=['parser', 'ner'])
nlp = spacy.load("en_core_web_sm",  disable=['parser', 'ner'])
data_lemmatized = lemmatization(data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

id2word = corpora.Dictionary(data_lemmatized)

texts = data_lemmatized
corpus = [id2word.doc2bow(text) for text in texts]

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, alpha='auto', num_topics=20, random_state=100,
                                           update_every=1, passes=20, per_word_topics=True)
print(lda_model.print_topics())
doc_lda  = lda_model[corpus]
print('\nPerplexity:', lda_model.log_perplexity(corpus))
coherence_model_lda = CoherenceModel(lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score:', coherence_lda)



for sent in nltk.sent_tokenize(clean_data):
    for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
         if hasattr(chunk, 'label'):
            print(f"{' '.join(c[0] for c in chunk):<35} {chunk.label()}")



# Entity extraction

import spacy.cli
spacy.cli.download("en_core_web_lg")
nlp = spacy.load("en_core_web_lg")
doc = nlp(clean_data)
for entities in doc.ents:
    print(f"{entities.text:<25} {entities.label_:<15} {spacy.explain(entities.label_)}")