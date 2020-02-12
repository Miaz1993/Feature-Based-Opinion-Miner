from All_Dictionary import nouns, negative_word, get_sentiment_dict
import pandas as pd
import re
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from LabelAlgo import CommentAnalyzer
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
from sklearn.metrics import accuracy_score

# Deep learning model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation,Dropout,Dense
from keras.layers import Flatten, LSTM, Input
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers.merge import Concatenate
from sklearn.preprocessing import MultiLabelBinarizer

if __name__ == '__main__':
  
  sentiment_dict = get_sentiment_dict('/home/cdsw/HAC/HAC/output_final.csv')

  ca = CommentAnalyzer(sentiment_dict,nouns,negative_word)
  # Read data
  engagement = pd.read_excel('Engagement2018.xlsx')
  engagement_eng=engagement[engagement['Language']==' English']
  engagement_eng=engagement_eng.dropna()
  final_list = []
  for sen in engagement_eng['Comment Text']:
    sent,_ = ca.preprocessing(sen)
    sent = ' '.join(sent)
    sentence_dic = ca.train_model_1(sen)  
    row_dict = sorted(sentence_dic.items(), key=lambda x: -abs(x[1]))
    for i in range(len(row_dict)):
      final_list.append([sent,row_dict[i][0],row_dict[i][1]])

                                         
  new_df = pd.DataFrame(final_list, columns = ['Text','Concern','Sentiment'])
  
  #new_df.to_csv('processed_df.csv')

  #labelling map
  #ps = PorterStemmer()
  #nouns_dict = new_df.Concern.value_counts().to_dict()
  #feature_cluster = {}
  #result = {}
  #finalmapping = {}
  #for key in nouns_dict:
  #  if ps.stem(key) in feature_cluster.keys():
  #    feature_cluster[ps.stem(key)].append(key)
  #  else:
  #    feature_cluster[ps.stem(key)] = [key]
  #  if ps.stem(key) in result:
  #    result[ps.stem(key)] = result[ps.stem(key)] + nouns_dict[key]
  #  else:
  #    result[ps.stem(key)] = nouns_dict[key]
  #for key in result:
  #    finalmapping[tuple(feature_cluster[key])] = key
    
  finalmap = {
   ('budget','cost','expense','costing'): 'budget&cost',
   ('change', 'changing', 'changed','transition'): 'change&transition',
   ('collaboration', 'collaborate','connect','cooperation','corporate','collaboration', 'collaborative', 'collaborating'): 'collaboration',
   ('communicative','communicates','communication','connect','communicated', 'communicate', 'communicating','information','info','resource','speak','convey','chance','interact','involvement'): 'communication&understanding',
   ('compensation','wage','bonus','paid','pay', 'paying','payment','payout','perk','payroll','pay', 'paying','payment','salary', 'salaried'): 'compensation',
   ('culture','environment','organization','organizational','organized'): 'culture&environment',
   ('development', 'developing','position','professional','career','synergy','opportunites','progress'): 'personal development',
   ('diversity', 'diverse','variety'): 'culture&diverse',
   ('effectiveness','efficiency', 'efficient','productivity','system','sytems'): 'effectiveness&quality',
   ('expertise','talent'): 'expertise&talent',
   ('team','teamwork','teambuilding','function', 'functional', 'functionality', 'functioning','group'): 'teamwork&functions',
   ('goal','objective','target', 'targeted', 'targeting','planning','focused', 'focusing'): 'goal&targeting',
   ('innovation', 'innovate', 'innovative', 'innovating','creativity','creation'): 'innovation&creativity',
   ('leader','leadership','leading','led','decision','decided','pioneer','president'): 'leadership&management',
   ('mbos', 'mbo'): 'mbo effectiveness',
   ('product','implementation','quality' ): 'implementation&product quality',
   ('project','timeline','premier'): 'work life balance',
   ('promotion', 'promoted', 'promoting',  'promotional', 'reward', 'rewarded', 'rewarding'): 'promotion&rewards',
   ('retail','price','consumer','customer','distributor','client'): 'retail&supply',
   ('roadblock','support','partnership','distribution','laborious','quantity','relaying','reliability','timeline','procedure'): 'support&efficiency',
   ('training', 'trained','coach','retraining'): 'training&coaching'}


  mappingConcerns = new_df.Concern.values

  mapping = []
  for i in mappingConcerns:
    for key in finalmap:
      if i in key:
        final = finalmap[key]
    mapping.append(final)
  
  new_df['Concern_exp'] = mapping
  new_df = new_df.groupby(['Text','Concern_exp']).min().reset_index()
  new_df['Sentiment'] = ['positive' if i >0 else 'negative' for i in new_df['Sentiment']]
  new_df['Sentiment'] = new_df['Concern_exp']+'_'+ new_df['Sentiment']
  new_df['Score'] = 1

  #unique_nouns = new_df.Concern.unique()
  #lemma_nouns_list = {}
  #for word in unique_nouns:
  #  if lemma.lemmatize(word) in lemma_nouns_list.keys():
  #new_df['Combine'] = new_df['Concern']+ ' : ' + new_df['Sentiment']
  #
  #new_df = new_df.groupby('Text')['Combine'].apply(','.join).reset_index()
  #
  #new_df['Label'] = [','.join(sorted(i.split(','))) for i in new_df['Combine']]
  #
  ##new_df['Label'] = new_df['Combine'].apply(lambda x: ','.join(sorted(x.split(','))))


  #21 classes
  pivot_df = pd.pivot_table(new_df,values = 'Score',index = ['Text'],columns = ['Sentiment'],aggfunc = np.min)
  pivot_df = pivot_df.reset_index()
  pivot_df = pivot_df.fillna(0)

  #subseting df
  map_all = {}
  for col in pivot_df.columns[1:]:
    val = pivot_df[col].value_counts().values[1]
    map_all[col] = val
  
  map_all = sorted(map_all.items(), key=lambda x: -abs(x[1]))
  
  sub_1 = ['teamwork&functions_negative','goal&targeting_negative',
           'personal development_negative','leadership&management_negative',
           'teamwork&functions_positive','goal&targeting_positive',
           'personal development_positive','leadership&management_positive']
  sub_2 = ['communication&understanding_negative','retail&supply_negative',
           'compensation_negative','promotion&rewards_negative',
           'mbo effectiveness_negative','communication&understanding_positive',
           'retail&supply_positive','compensation_positive',
           'promotion&rewards_positive','mbo effectiveness_positive']
  sub_3 = ['budget&cost_negative','implementation&product quality_negative',
           'effectiveness&quality_negative','implementation&product quality_positive',
           'training&coaching_negative','budget&cost_positive',
           'effectiveness&quality_positive','training&coaching_positive']
  sub_4 = ['innovation&creativity_negative','innovation&creativity_positive',
           'work life balance_negative','culture&environment_negative',
           'expertise&talent_negative','work life balance_positive',
           'culture&environment_positive','expertise&talent_positive']
  sub_5 = ['collaboration_negative','culture&diverse_negative',
           'support&efficiency_negative','change&transition_negative',
           'change&transition_positive','support&efficiency_positive',
           'collaboration_positive','culture&diverse_positive']
  
  
  #subseting
  #sub1
  pivot_sub1 = pivot_df[['Text']+ sub_1]
  pivot_sub1['sumcol'] = pivot_df[sub_1].sum(axis = 1)
  pivot_sub1 = pivot_sub1[pivot_sub1.sumcol != 0]
  
  #sub2
  pivot_sub2 = pivot_df[['Text']+ sub_2]
  pivot_sub2['sumcol'] = pivot_df[sub_2].sum(axis = 1)
  pivot_sub2 = pivot_sub2[pivot_sub2.sumcol != 0]  
  
  #sub3
  pivot_sub3 = pivot_df[['Text']+ sub_3]
  pivot_sub3['sumcol'] = pivot_df[sub_3].sum(axis = 1)
  pivot_sub3 = pivot_sub3[pivot_sub3.sumcol != 0]   

  #sub4
  pivot_sub4 = pivot_df[['Text']+ sub_4]
  pivot_sub4['sumcol'] = pivot_df[sub_4].sum(axis = 1)
  pivot_sub4 = pivot_sub4[pivot_sub4.sumcol != 0]   
  
  #sub5
  pivot_sub5 = pivot_df[['Text']+ sub_5]
  pivot_sub5['sumcol'] = pivot_df[sub_5].sum(axis = 1)
  pivot_sub5 = pivot_sub4[pivot_sub5.sumcol != 0] 
  
  
  
  
  # Build up Multilabel classifier
  labels=new_df['Sentiment'].unique()
  train, test=train_test_split(pivot_df,random_state=13, test_size=0.33,shuffle=True)
  X_train=train['Text']
  X_test=test['Text']
  print(X_train.shape)
  print(X_test.shape)

  # Random Forest
  from sklearn.metrics import classification_report
  from sklearn.metrics import precision_recall_fscore_support as score

  stop=set(stopwords.words('english'))
  
  
  def ClassifierEval(clf,feature):
    evaluation_list = []
    exp = []
    clf_pipeline=Pipeline([
      ('tfidf',feature),
      ('clf',OneVsRestClassifier(clf))])
    for label in labels:
      print('...Processing {}'.format(label))
      clf_pipeline.fit(X_train,train[label])
      pred=clf_pipeline.predict(X_test)
      print(label,'accuracy')
      accuracy=accuracy_score(test[label],pred)
      print(accuracy) 
      precision,recall,fscore,support=score(test[label],pred)
      if len(precision)>1:
        exp.append([label,accuracy,precision[1],recall[1],fscore[1]])
      evaluation_list.append([label,accuracy,precision[0],recall[0],fscore[0]])  
    exp_df = pd.DataFrame(exp, columns = ['Label','Accuracy','Precision','Recall','Fscore'])
    acc_df = pd.DataFrame(evaluation_list, columns = ['Label','Accuracy','Precision','Recall','Fscore'])
    
    return exp_df, acc_df
  
  
  # Different classifier - OneVsRest & TFIDF
  tfidf = TfidfVectorizer(stop_words=stop)
  
  #random forest
  rf = RandomForestClassifier(random_state = 0)
  exp,acc = ClassifierEval(rf,tfidf)
  
  #linearSVC
  svc = LinearSVC(random_state = 0)
  exp,acc = ClassifierEval(rf,tfidf)
  
  #Logstic Regression
  lr = LogisticRegression(solver='sag',random_state = 0)
  exp,acc = ClassifierEval(lr,tfidf)
  
  #Naive Bayes
  nb = GaussianNB(random_state = 0)
  exp,acc = ClassifierEval(nb,tfidf)
  
  
  #Different classifier - OneVsRest & Word2Vec

  #word2vec with one/vsall
  
  from gensim.models import Word2Vec
  Vocab_list = (pivot_df['Text'].apply(lambda x:str(x).strip().split()))
  models = Word2Vec(Vocab_list, size = 50)
  WordVectorz = dict(zip(models.wv.index2word,models.wv.vectors))

  class AverageEmbeddingVectorizer(object):
    def __init__(self, word2vec):
      self.word2vec = word2vec
      self.dim = 50

    def fit(self, X, y):
      return self

    def transform(self, X):
      return np.array([
        np.sum([self.word2vec[w] for w in words if w in self.word2vec]
                      or [np.zeros(self.dim)],axis = 0) for words in X])
  
  w2v = AverageEmbeddingVectorizer(WordVectorz)
  
  #random forest
  rf = RandomForestClassifier(random_state = 0)
  exp,acc = ClassifierEval(rf,w2v)
  
  #linearSVC
  svc = LinearSVC(random_state = 0)
  exp,acc = ClassifierEval(rf,w2v)
  
  #Logstic Regression
  lr = LogisticRegression(solver='sag',random_state = 0)
  exp,acc = ClassifierEval(lr,w2v)
  
  #Naive Bayes
  nb = GaussianNB(random_state = 0)
  exp,acc = ClassifierEval(nb,w2v)


  #Deep Learning - keras
  tokenizer=Tokenizer(num_words=500,lower=True)
  tokenizer.fit_on_texts(X_train)

  X_train_keras=tokenizer.texts_to_sequences(X_train)
  X_test_keras=tokenizer.texts_to_sequences(X_test)


  vocab_size=len(tokenizer.word_index)+1
  maxlen=1
  max_features=500
  embed_size=1

  X_train_keras=pad_sequences(X_train_keras,padding='post',maxlen=maxlen)
  X_test_keras=pad_sequences(X_test_keras,padding='post',maxlen=maxlen)

  input1=Input(shape=(maxlen,))
  embedding = Embedding(max_features,embed_size)(input1)
  x=LSTM(50,return_sequences=True)(embedding)
  pool=GlobalMaxPooling1D()(x)
  dropout1=Dropout(0.1)(pool)
  dense1=Dense(50,activation='relu')(dropout1)
  dropout2=Dropout(0.1)(dense1)
  dense2=Dense(1,activation='sigmoid')(dropout2)
  model=Model(input=input1,output=dense2)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#  print(model.summary())
  
  evaluation_list = []
  for label in labels:
    history=model.fit(X_train_keras,train[label],batch_size=16,epochs=30)
    prediction=model.predict(X_test_keras)
    score=model.evaluate(X_test_keras,test[label])
    accuracy= history.history['accuracy'][49]
    #accuracy=accuracy_score(test[label],prediction)
    #precision,recall,fscore,support=score(prediction,test[label])
    evaluation_list.append([label,accuracy])  
  acc_df = pd.DataFrame(evaluation_list, columns = ['Label','Accuracy_keras'])
  acc_df

  acc_df.plot.line(x='Label',y=['Accuracy_keras'],color='red',legend=False)
  #  print('Test Score:',score[0])
  #  print('Test Accuray',score[1])



  # Simple Transformer BERT
  from simpletransformers.classification import MultiLabelClassificationModel
  
  def transform_df(pivot_sub):
    values=pivot_sub.drop(['Text','sumcol'],axis=1)
    pivot_sub['Text']=pivot_sub['Text'].apply(lambda z:z.replace('\n',''))
    data={'Text':pivot_sub['Text'],
         'Labels':list(zip(values.values.tolist()))}
    df=pd.DataFrame(data,columns=['Text','Label'])
    df['Label']=values.values.tolist()
    return df
  
  
  def bert_based_model(df,num):
    df=df.sample(n=len(df),random_state=42).reset_index(drop=True)
    train, test=train_test_split(df,random_state=13, test_size=0.1,shuffle=True)
    #num=int(len(train['Label'][1]))
    # roberta model
#    roberta = MultiLabelClassificationModel('roberta','roberta-base',num_labels = num,
#                                            args = {'output_dir':'outputs_roberta/','train_batch_size':16,'gradient_accumulation_steps':5,
#                                                    'learning_rate':0.00005,'num_train_epochs':5,
#                                                    'max_seq_length':512,'overwrite_output_dir':True},
#                                            use_cuda=False)
#    roberta.train_model(train)
#    result_r,model_outputs_r,wrong_predictions_r=roberta.eval_model(test)
#    
#    
#    # Albert model
#    albert = MultiLabelClassificationModel('albert','albert-base-v1',num_labels = num,
#                                        args = {'output_dir':'outputs_albert/','train_batch_size':16,'gradient_accumulation_steps':5,
#                                                'learning_rate':0.00005,'num_train_epochs':5,
#                                                'max_seq_length':512,'overwrite_output_dir':True},
#                                           use_cuda=False)
#
#    albert.train_model(train)
#    result_a,model_outputs_a,wrong_predictions_a=albert.eval_model(test)
    
    # Distilbert model
    distilbert = MultiLabelClassificationModel('distilbert','distilbert-base-uncased',num_labels = num,
                                        args = {'output_dir':'outputs_disbert/',
                                                'train_batch_size':16,
                                                'gradient_accumulation_steps':5,
                                                'learning_rate':0.0005,
                                                'num_train_epochs':5,
                                                'max_seq_length':512,
                                                'overwrite_output_dir':True},
                                               use_cuda=False)

    distilbert.train_model(train)
    #default evaluation
    result_d,model_outputs_d,wrong_prediction_d=distilbert.eval_model(test)
#    accy = {result_d_acc,model_outputs_d_acc,wrong_predictions_d_acc}
#    data = {'Roberta':result_r,'Albert':result_a,'Distilbert':result_d}
#    model_eval = pd.DataFrame(data)

    return result_d,model_outputs_d,wrong_prediction_d
  
  # result for sub1
  df1=transform_df(pivot_sub1)
  a,b,c=bert_based_model(df1,8)
  
  ##loss
  thres = [0.4,0.5,0.6,0.7,0.8,0.9]
  
  ##????????
  m=MultiLabelClassificationModel('distilbert','/home/cdsw/outputs_disbert/')
  
  
  
  
  
  
  
  
  #result for sub2
  df2=transform_df(pivot_sub2)
  eval2=bert_based_model(df2,10)

  #result for sub3
  df3=transform_df(pivot_sub3)
  eval3=bert_based_model(df3,8)
  
  #result for sub4
  df4=transform_df(pivot_sub4)
  eval4=bert_based_model(df4,8)
  
  #result for sub4
  df5=transform_df(pivot_sub5)
  eval5=bert_based_model(df5)
  
  
