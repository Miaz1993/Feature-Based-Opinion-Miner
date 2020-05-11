from nltk.util import ngrams
from collections import Counter
import itertools
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk import ngrams
import re

stop_words = set(stopwords.words('english'))
#bi&trigram
def preprocessing(sentence):
  sentence = re.sub('[^A-Za-z0-9\s]+', '', sentence).lower()
  sentence = word_tokenize(sentence)
  words = [w for w in sentence if not w in stop_words]
  lemma=WordNetLemmatizer()
  sentence_raw=[lemma.lemmatize(te) for te in words]
#  print(sentence_raw)
  return sentence_raw

def get_ngrams(text, n):
  n_grams = ngrams(text,n)
#  for grams in n_grams:
#    print(grams)
#    print(' '.join(grams))
  return [' '.join(grams) for grams in n_grams]

def gramfreq(text,n,num):
#  print(len(text))
  result = get_ngrams(text,n)
  result_count = Counter(result)
  df = pd.DataFrame.from_dict(result_count, orient = 'index')
  df = df.rename(columns = {'index':'words', 0:'frequency'})
  return df.sort_values(['frequency'],ascending = [0])[:num]

def gram_table(gram, length,text):
  out = pd.DataFrame(index = None)
  for i in gram:
    process = [preprocessing(sen) for sen in text.values]
    merged = list(itertools.chain(*process))
    table = pd.DataFrame(gramfreq(merged,i,length).reset_index())
    table.columns = ["{}-Gram".format(i),'Occurence']
    out = pd.concat([out,table],axis = 1)
  return out

# Read data
#engagement = pd.read_excel('Engagement2018.xlsx')
#engagement.head()
#engagement['Language'].value_counts()
#engagement_eng=engagement[engagement['Language']==' English']
#engagement_eng.info()
#engagement_eng=engagement_eng.dropna()
#
#gram_table(gram = [2,3,4],length=50,text = engagement_eng['Comment Text'])

#for i in ngrams([['i','am', 'very', 'good'],["i","is","a","bug"]],2):
#  print(i)