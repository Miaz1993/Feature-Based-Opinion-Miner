from All_Dictionary import nouns, negative_word, get_sentiment_dict
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class CommentAnalyzer:
  def __init__(self,sentiment_dict, keywords,negative_word):
    self._sentiment_dict = sentiment_dict
    self._keywords = keywords
    self._negative_word = negative_word
  
  def preprocessing(self, sentence):
    """
    sentence precrocessing to get 
    rid of redundant words
    """
    
    sentence = re.sub('[^A-Za-z0-9\s]+', '', sentence).lower()
    sentence = word_tokenize(sentence)
    lemma=WordNetLemmatizer()
    sentence_raw=[lemma.lemmatize(te) for te in sentence]
    
    new_sentence = []
    for word in sentence_raw:
      if word in self._sentiment_dict.keys():
        word_object = word_obj(word,'sen',self._sentiment_dict[word])
      elif word in self._negative_word:
        word_object = word_obj(word,'neg',-1)
      elif word in self._keywords:
        word_object = word_obj(word,'key',0)
      else:
        word_object = word_obj(word)
      new_sentence.append(word_object)
      
    return sentence_raw,new_sentence
  
  @staticmethod
  def add_sentiment(sen_dic, keyword, sentiment_word, neg_sent = 1, value_flag = False):
    if not value_flag: 
      score = sentiment_word._score
    else:
      score = sentiment_word
    if keyword._word in sen_dic:
      sen_dic[keyword._word] += neg_sent*score
    else:
      sen_dic[keyword._word] = neg_sent*score
    return sen_dic
  
  @staticmethod
  def sentiment_scores(sentence):
    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    if sentiment_dict['compound']>=0.05:
      return 1
    elif sentiment_dict['compound']<=-0.05:
      return -1
    else:
      return 0
      
  def train_model_1(self, sentence):
    sentence_raw, sentence = self.preprocessing(sentence)
    sen_dic = {}
    for i,word in enumerate(sentence):
      if word._word_type == 'key':
        # search backward if there is a sentiment word 
        if i-1>=0 and sentence[i-1]._word_type == 'sen':
          # search if there is a negative word
          for k in range(1,4):
            if i-1-k >=0 and sentence[i-1-k]._word_type == 'neg':
              neg_sent = -1
            else:
              neg_sent = 1
          sen_dic = self.add_sentiment(sen_dic, word, sentence[i-1], neg_sent)
          # if no sentiment word search if there is a negative word
        else:
          neg_index = -1
          for k in range(1,3):
            if i-k>=0 and sentence[i-k]._word_type == 'neg':
              neg_index = i-k
          if neg_index >= 0:
            sen_dic = self.add_sentiment(sen_dic, word, sentence[neg_index])
          # search forward 
          else:
            sen_index = -1
            for k in range(1,4):
              if i+k < len(sentence) and sentence[i+k]._word_type == 'sen':
                # only label the latest index
                sen_index = i+k
            if sen_index >=0 : # sucessfully labeled
              if sentence[sen_index-1]._word_type == 'neg':
                neg_sent = -1
              else:
                neg_sent = 1
              sen_dic = self.add_sentiment(sen_dic, word, sentence[sen_index], neg_sent)
            else: # not label sentiment by above methods(B&F) then use VADER
              if not sen_dic: 
                start_index = 0
              else:
                start_index = i+1
                
              for k in range(i+1, len(sentence)):
                if sentence[k]._word_type == 'key':
                  break
              end_index = min(k,len(sentence)-1)
              sen_vader = self.sentiment_scores(' '.join(sentence_raw[start_index:end_index+1]))
              sen_dic = self.add_sentiment(sen_dic, word, sen_vader, value_flag = True)
                                                
    return sen_dic              
              
                    

class word_obj:
  def __init__(self, word, word_type = None, score = 0):
    self._word = word
    self._word_type = word_type
    self._score = score

#if __name__ == '__main__':
#  sentiment_dict = get_sentiment_dict('/home/cdsw/HAC/HAC/output_final.csv')
#
#  ca = CommentAnalyzer(sentiment_dict,nouns,negative_word)
#
#
#  # Read data
#  engagement = pd.read_excel('Engagement2018.xlsx')
#  engagement_eng=engagement[engagement['Language']==' English']
#  engagement_eng=engagement_eng.dropna()
#
#  sen = engagement_eng['Comment Text'][2]
#  _,a = ca.preprocessing(sen)
#  for i in a:
#    print(i._word,i._word_type,i._score)
#  b = ca.train_model_1(sen)  
#  print(b)                                              
  




  
  
  
  
  
  
  


      
