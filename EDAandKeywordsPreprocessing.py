import pandas as pd
import re
import string
import nltk as nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk import ngrams
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import gensim
from gensim import corpora,models
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models import Word2Vec
import multiprocessing
import pyLDAvis.gensim
from pprint import pprint
from glove import Glove,Corpus
from wordcloud import WordCloud
from functools import reduce
from summa import keywords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# Read data
engagement = pd.read_excel('Engagement2018.xlsx')
engagement.head()
engagement['Language'].value_counts()
engagement_eng=engagement[engagement['Language']==' English']
engagement_eng.info()
engagement_eng=engagement_eng.dropna()

# Clean data
engagement_eng['Comment Clean']=engagement_eng['Comment Text'].str.replace(r'[^A-Za-z0-9\s]+','')

stop=set(stopwords.words('english'))
exclude=set(string.punctuation)
lemma=WordNetLemmatizer()


def clean(doc):
  stop_free=' '.join([i for i in doc.lower().split() if i not in stop])
  is_english = [i for i in word_tokenize(stop_free) if i.isalpha()]
  punc_free=' '.join(ch for ch in is_english if ch not in exclude)
  normalized=' '.join(lemma.lemmatize(word) for word in punc_free.split())
  normal=normalized.encode('ascii','ignore').decode('ascii')
  return normal

clean=[clean(doc).split() for doc in engagement_eng['Comment Clean']]
text=[' '.join(i) for i in clean]
df=pd.DataFrame(engagement_eng['Comment Text'])

# EDA
def wordfreq(words,z):
  word_list=nltk.FreqDist(words)
  topN=z
  result = pd.DataFrame(word_list.most_common(topN),
                        columns=['Word','Frequency']).set_index('Word')
  result.plot.bar(rot=1,color='blue',figsize=(15,7))
  plt.xticks()

words=reduce(lambda x,y:x+y,clean)
wordfreq(words,10)

# Sentence length distribution
def sentence_length_plot(doc):
  sentence_length=[len(tokens) for tokens in doc]
  fig=plt.figure(figsize=(10,10))
  plt.xlabel('Sentence Length')
  plt.ylabel('Number of sentences')
  plt.hist(sentence_length)
  plt.show()
sentence_length_plot(clean)

sentence_length=[len(tokens) for tokens in clean]
len([i for i in sentence_length if i > 100])

#LDA Model - Select key concerns
dictionary=corpora.Dictionary(clean)
dictionary.filter_extremes(no_below = 100, no_above = 2, keep_n = 10000)
doc_term_matrix=[dictionary.doc2bow(doc) for doc in clean]
tfidf = models.TfidfModel(doc_term_matrix)
corpus_tfidf=tfidf[doc_term_matrix]

ldamodel=LdaMulticore(corpus_tfidf,num_topics=7,id2word=dictionary,passes=100)
print(*ldamodel.print_topics(num_topics=7,num_words=20),sep='\n')
lda_display=pyLDAvis.gensim.prepare(ldamodel,doc_term_matrix,dictionary,sort_topics=False)

# Other way to select key concerns
concerns=[keywords.keywords(i).replace('\n',',').split(',') for i in df['Comment Text']]
concern=reduce(lambda x,y:x+y,concerns)
wordfreq(concern,40)


# Build up dictionary
keywords={'bonus',
 'business',
 'career',
 'change',
 'collaboration',
 'communication',
 'consumer',
 'cost',
 'customer',
 'decision',
 'development',
 'distributor',
 'diversity',
 'efficiency',
 'focus',
 'function',
 'goal',
 'group',
 'information',
 'innovation',
 'leadership',
 'mbo',
 'mbos',
 'mgt',
 'opportunity',
 'organization',
 'pay',
 'performance',
 'product',
 'project',
 'promotion',
 'quality',
 'resource',
 'reward',
 'role',
 'salary',
 'system',
 'target',
 'team',
 'teamwork',
 'training'}

# Use word embedding find similar words
EMB_DIM = 50
w2v = Word2Vec(df['Comment Text'],size=EMB_DIM,min_count=10,window=2,workers=multiprocessing.cpu_count())
vectors=w2v.wv
[vectors.similar_by_word(word) for word in keywords]

