from All_Dictionary import nouns, negative_word, get_sentiment_dict
import pandas as pd
import numpy as np
import re

#nltk
from nltk.tokenize import RegexpTokenizer,word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from LabelAlgo import CommentAnalyzer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from nltk.corpus import stopwords

#gensim
import multiprocessing
import pyLDAvis.gensim
from gensim import corpora,models
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.doc2vec import Doc2Vec,TaggedDocument
from gensim.models import Word2Vec
from pprint import pprint
from glove import Glove,Corpus
from gensim.models import Word2Vec

#vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.metrics import accuracy_score,precision_score
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from nltk.stem import PorterStemmer
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

#tools
import itertools
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import BiTriGram as btg

if __name__ == '__main__':
  sentiment_dict = get_sentiment_dict('/home/cdsw/HAC/HAC/output_final.csv')
  del sentiment_dict['system']
  del sentiment_dict['quality']
  del sentiment_dict['decision']
  
  ca = CommentAnalyzer(sentiment_dict,nouns,negative_word)
  # Read data
  engagement = pd.read_excel('filename.xlsx')
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
#  ps = PorterStemmer()
#  nouns_dict = new_df.Concern.value_counts().to_dict()
#  feature_cluster = {}
#  result = {}
#  finalmapping = {}
#  for key in nouns_dict:
#    if ps.stem(key) in feature_cluster.keys():
#      feature_cluster[ps.stem(key)].append(key)
#    else:
#      feature_cluster[ps.stem(key)] = [key]
#    if ps.stem(key) in result:
#      result[ps.stem(key)] = result[ps.stem(key)] + nouns_dict[key]
#    else:
#      result[ps.stem(key)] = nouns_dict[key]
#  for key in result:
#      finalmapping[tuple(feature_cluster[key])] = key

# finalmap load 
  finalmap= json.loads('finalmap.json')
# example
#   finalmap = {
#    ('budget','cost',..,'expense'): 'budget&cost',
#    ('collaboration',...,'cooperation','collaboration'): 'collaboration',
#    ('communication','communicating',....,'information','info','resource'): 'communication&understanding',
#    ('compensation','wage','bonus','payment',...,'perk','salary', 'salaried'): 'compensation',
#    ('expertise',...,'talent'): 'expertise&talent',
#    ('team','teamwork',...,'function','functionality', 'group'): 'teamwork&functions',
#    ('goal','target', 'targeted', 'targeting','objective','focused', 'focusing'): 'goal&targeting',
#    ('innovation',...,'innovating','creativity'): 'innovation&creativity',
#    ('mbos', 'mbo','system'): 'mbo&system'}
#......
      
  mappingConcerns = new_df.Concern.values

# map key to value and save back to dataframe
  mapping = []
  for i in mappingConcerns:
    for key in finalmap:
      if i in key:
        final = finalmap[key]
    mapping.append(final)

  new_df['Concern_exp'] = mapping
  new_df = new_df.groupby(['Text','Concern_exp']).min().reset_index()
  
  # df for ml
  new_df['Sentiment_ml'] = [1 if i>0 else -1 if i<0 else 0 for i in new_df['Sentiment']]
  pivot = pd.pivot_table(new_df,values = 'Sentiment_ml',index = ['Text'],columns = ['Sentiment'],aggfunc = np.min)
  pivot_df = pivot_df.reset_index()
  pivot_df = pivot_df.fillna(0)  
  
  # Build up Multilabel classifier
  labels=new_df['Sentiment'].unique()
  train, test=train_test_split(pivot,random_state=23, test_size=0.33,shuffle=True)
  X_train=train['Text']
  X_test=test['Text']

  # machine learning models
  from sklearn.metrics import classification_report
  from sklearn.metrics import precision_recall_fscore_support as score
  
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
    return exp_df,acc_df
  
  # OneVsRest & TFIDF
  tfidf = TfidfVectorizer(stop_words=stop)
  
  #random forest
  rf = RandomForestClassifier(random_state = 0)
  exp,acc= ClassifierEval(rf,tfidf)
  np.mean(exp['Precision']) #0.274
  
  #gbdt
  gbdt = GradientBoostingClassifier(random_state=0)
  exp,acc = ClassifierEval(gbdt,tfidf)
  np.mean(exp['Precision']) #0.243
  
  #SVC-kernel
  svc = SVC(kernel = 'rbf', C = 0.1, gamma = 0.1)
  exp,acc = ClassifierEval(svc,tfidf)
  np.mean(exp['Precision']) #0.271
  
  #Logstic Regression
  lr = LogisticRegression(solver='sag',random_state = 0)
  exp,acc = ClassifierEval(lr,tfidf)
  np.mean(exp['Precision']) #0.249
  
  #SGD
  sgd = SGDClassifier(loss="perceptron", eta0=0.1, learning_rate="constant")
  exp,acc = ClassifierEval(sgd,tfidf)
  np.mean(exp['Precision']) #0.254
  
  #OneVsRest & Word2Vec
  #word2vec with one/vsall
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
  rf = RandomForestClassifier(random_state = 0，n_estimators=150, max_depth=15)
  exp,acc = ClassifierEval(rf,w2v)
  np.mean(exp['Precision']) #0.079
  
  #linearSVC
  svc = LinearSVC(random_state = 0)
  exp,acc = ClassifierEval(rf,w2v)
  np.mean(exp['Precision']) #0.079
  
  #Logstic Regression
  lr = LogisticRegression()
  exp,acc = ClassifierEval(lr,w2v)
  np.mean(exp['Precision']) #0.0
  
  #Naive Bayes
  nb = MultinomialNB()
  exp,acc = ClassifierEval(nb,w2v)
  np.mean(exp['Precision']) #0.0
  
  #SGD
  sgd = SGDClassifier(loss="perceptron", eta0=0.1, learning_rate="constant")
  exp,acc = ClassifierEval(sgd,w2v)
  np.mean(exp['Precision']) #0.092

  #deep learning modelling
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

  #precision
  from keras import backend as K
  def precision_m(y_true,y_pred):
    true_positive=K.sum(K.round(K.clip(y_true*y_pred,0,1)))
    predicted_positives=K.sum(K.round(K.clip(y_pred,0,1)))
    precision=true_positive/(predicted_positives+K.epsilon())
    return precision
  
  input1=Input(shape=(maxlen,))
  embedding = Embedding(max_features,embed_size)(input1)
  x=LSTM(50,return_sequences=True)(embedding)
  pool=GlobalMaxPooling1D()(x)
  dropout1=Dropout(0.1)(pool)
  dense1=Dense(50,activation='relu')(dropout1)
  dropout2=Dropout(0.1)(dense1)
  dense2=Dense(1,activation='sigmoid')(dropout2)
  model=Model(input=input1,output=dense2)
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy',precision_m])
  print(model.summary())
  
  evaluation_list = []
  for label in labels:
    history=model.fit(X_train_keras,train[label],batch_size=16,epochs=30)
    prediction=model.predict(X_test_keras)
    loss,acc,pre=model.evaluate(X_test_keras,test[label])
    #accuracy= history.history['accuracy'][29]
    #accuracy=accuracy_score(test[label],prediction)
    evaluation_list.append([label,acc,pre])  
  acc_df = pd.DataFrame(evaluation_list, columns = ['Label','Accuracy_keras','Precision_keras'])
  np.mean(acc_df['Precision_keras']) #0.012

  acc_df.plot.line(x='Label',y=['Accuracy_keras'],color='red',legend=False)
  #  print('Test Score:',score[0])
  #  print('Test Accuray',score[1])

  # Simple Transformer BERT
  from simpletransformers.classification import MultiLabelClassificationModel
  
  # check labels' correlation
  # subset dataset to decrease sparsity
  def cramers_corrected_stat(confusion_matrix):
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0,phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
  
  cols = ['budget&cost','change&transition','collaboration','communication&understanding',
          'compensation','culture&environment','diversity','effectiveness','expertise&talent',
          'goal&targeting','innovation&creativity','leadership&management','mbo&system',
          'personal development','product&project','promotion&rewards','retail&supply',
          'support&resource','teamwork&functions','timeline','training&coaching']
  
  corrM = np.zeros((len(cols),len(cols)))
  
  for col1,col2 in itertools.combinations(cols,2):
    idx1,idx2 = cols.index(col1),cols.index(col2)
    corrM[idx1,idx2] = cramers_corrected_stat(pd.crosstab(chi_df[col1],chi_df[col2]))
    corrM[idx2,idx1] = corrM[idx1,idx2]
  
  corr = pd.DataFrame(corrM,index = cols, columns = cols)
  fig,ax = plt.subplots(figsize=(15,15))
  cmap = sns.diverging_palette(220,20, sep = 20,as_cmap = True)
  ax = sns.heatmap(corr,cmap = cmap)
  ax.set_title('Cramer V Correlation between Variables')
  fig.savefig('cramers v')
  
  # 42 classes -- simple transformers
  #data transform for transfer learning

  new_df['Sentiment_raw'] = ['positive' if i >0 else 'negative' for i in new_df['Sentiment']]
  new_df['Sentiment'] = new_df['Concern_exp']+'_'+ new_df['Sentiment_raw']
  new_df['Score'] = 1

  pivot_df = pd.pivot_table(new_df,values = 'Score',index = ['Text'],columns = ['Sentiment'],aggfunc = np.min)
  pivot_df = pivot_df.reset_index()
  pivot_df = pivot_df.fillna(0)  
  
  def subset(pivot_df, sub):
  	pivot_sub = pivot_df[['Text']+ sub]
  	pivot_sub['sumcol'] = pivot_df[sub].sum(axis = 1)
  	pivot_sub = pivot_sub[pivot_sub.sumcol != 0]
  	return pivot_sub
  	
  def transform_df(pivot_sub):
    values=pivot_sub.drop(['Text','sumcol'],axis=1)
    pivot_sub['Text']=pivot_sub['Text'].apply(lambda z:z.replace('\n',''))
    data={'Text':pivot_sub['Text'],
         'Labels':list(zip(values.values.tolist()))}
    df=pd.DataFrame(data,columns=['Text','Label'])
    df['Label']=values.values.tolist()
    return df
  
  sub_1 = ['teamwork&functions_negative','teamwork&functions_positive',
          'communication&understanding_negative','communication&understanding_positive',
          'effectiveness_negative','effectiveness_positive',
           'collaboration_negative', 'collaboration_positive',
           'mbo&system_negative','mbo&system_positive']
  
  sub_2 = ['goal&targeting_negative','goal&targeting_positive',
           'personal development_negative','personal development_positive',
          'promotion&rewards_positive','promotion&rewards_negative',
           'compensation_negative','compensation_positive']
  
  sub_3 = ['leadership&management_negative','leadership&management_positive',
           'training&coaching_negative','training&coaching_positive',
           'expertise&talent_negative','expertise&talent_positive',
           'support&resource_negative','support&resource_positive',
           'change&transition_negative','change&transition_positive']

  sub_4 = ['product&project_positive','product&project_negative',
          'budget&cost_negative','budget&cost_positive',
          'retail&supply_negative','retail&supply_positive',
          'innovation&creativity_negative','innovation&creativity_positive']
  
  sub_5 = ['timeline_negative','culture&environment_negative',
           'timeline_positive','culture&environment_positive',
            'diversity_negative','diversity_positive']

  #subseting
  pivot_sub1 = subset(pivot_df,sub_1)
  pivot_sub2 = subset(pivot_df,sub_2)
  pivot_sub3 = subset(pivot_df,sub_3)
  pivot_sub4 = subset(pivot_df,sub_4) 
  pivot_sub5 = subset(pivot_df,sub_5) 

  # result for sub1
  df1=transform_df(pivot_sub1)
  df1=df1.sample(n=len(df1),random_state=42).reset_index(drop=True)
  train1, test1=train_test_split(df1,random_state=13,test_size=0.1,shuffle=True)
  distilbert1 = MultiLabelClassificationModel('distilbert','distilbert-base-uncased',num_labels = 10,
                                        args = {'output_dir':'outputs_disbert1/',
                                                'train_batch_size':8,
                                                'gradient_accumulation_steps':5,
                                                'learning_rate':0.00005,
                                                'num_train_epochs':8,
                                                'max_seq_length':512,
                                                'overwrite_output_dir':True},
                                               use_cuda=False)
  distilbert1.train_model(train1)
  result1,model_outputs1,wrong_prediction1=distilbert1.eval_model(test1)
  
  #result for sub2
  df2=transform_df(pivot_sub2)
  df2=df2.sample(n=len(df2),random_state=42).reset_index(drop=True)
  train2, test2=train_test_split(df2,random_state=13,test_size=0.1,shuffle=True)
  distilbert2 = MultiLabelClassificationModel('distilbert','distilbert-base-uncased',num_labels = 8,
                                        args = {'output_dir':'outputs_disbert2/',
                                                'train_batch_size':8,
                                                'gradient_accumulation_steps':5,
                                                'learning_rate':0.00005,
                                                'num_train_epochs':8,
                                                'max_seq_length':512,
                                                'overwrite_output_dir':True},
                                               use_cuda=False)
  distilbert2.train_model(train2)
  result2,model_outputs2,wrong_prediction2=distilbert2.eval_model(test2)

  #result for sub3
  df3=transform_df(pivot_sub3)
  df3=df3.sample(n=len(df3),random_state=42).reset_index(drop=True)
  train3, test3=train_test_split(df3,random_state=13,test_size=0.1,shuffle=True)
  distilbert3 = MultiLabelClassificationModel('distilbert','distilbert-base-uncased',num_labels = 10,
                                        args = {'output_dir':'outputs_disbert3/',
                                                'train_batch_size':8,
                                                'gradient_accumulation_steps':5,
                                                'learning_rate':0.0003,
                                                'num_train_epochs':8,
                                                'max_seq_length':512,
                                                'overwrite_output_dir':True},
                                               use_cuda=False)
  distilbert3.train_model(train3)
  result3, model_outputs3, wrong_prediction3 = distilbert3.eval_model(test3)
  
  #result for sub4
  df4=transform_df(pivot_sub4)
  df4=df4.sample(n=len(df4),random_state=42).reset_index(drop=True)
  train4, test4=train_test_split(df4,random_state=13,test_size=0.1,shuffle=True)
  distilbert4 = MultiLabelClassificationModel('distilbert','distilbert-base-uncased',num_labels = 8,
                                        args = {'output_dir':'outputs_disbert4/',
                                                'train_batch_size':,8
                                                'gradient_accumulation_steps':5,
                                                'learning_rate':0.0003,
                                                'num_train_epochs':8,
                                                'max_seq_length':512,
                                                'overwrite_output_dir':True},
                                               use_cuda=False)
  distilbert4.train_model(train4)
  result4, model_outputs4, wrong_prediction4 = distilbert4.eval_model(test4)
  
  #result for sub5
  df5=transform_df(pivot_sub5)
  df5=df5.sample(n=len(df5),random_state=42).reset_index(drop=True)
  train5, test5=train_test_split(df5,random_state=13,test_size=0.1,shuffle=True)
  distilbert5 = MultiLabelClassificationModel('distilbert','distilbert-base-uncased',num_labels = 6,
                                        args = {'output_dir':'outputs_disbert5/',
                                                'train_batch_size':8,
                                                'gradient_accumulation_steps':5,
                                                'learning_rate':0.0002,
                                                'num_train_epochs':8,
                                                'max_seq_length':512,
                                                'overwrite_output_dir':True,
                                               'fp16':False},
                                             use_cuda=False)
  distilbert5.train_model(train5)
  result5, model_outputs5, wrong_prediction5 = distilbert5.eval_model(test5)  
   
   
  #final report data dashboard
  #dashboard
  sns.set(style = 'whitegrid')
  fig,ax = plt.subplots()
  fig.set_size_inches(20,15)
  #by sentiment
  concern_count = new_df.Sentiment_raw.value_counts()
  sns.countplot(y = 'Sentiment_raw',data = new_df,palette = 'vlag',saturation = 0.75)
  
  #By concern
  concern_count = new_df.Concern_exp.value_counts()
  sns.countplot(y = 'Concern_exp',data = new_df,palette = 'Blues',saturation = 0.75)
  
  #second layer reasoning
  sns.countplot(y = 'Concern_exp',hue = 'Sentiment_raw',data = new_df, palette = 'vlag')
  
  #Third layer reasoning
  gram_analysis1 = new_df[new_df.Sentiment == 'timeline_negative']['Text']
  gram_analysis2 = new_df[new_df.Sentiment =='culture&environment_negative']['Text']
  gram_analysis3 = new_df[new_df.Sentiment=='compensation_negative']['Text']
  gram_analysis4 = new_df[new_df.Sentiment=='personal development_negative']['Text']
  btg.gram_table(gram = [2,3,4],length=10,text = gram_analysis1)
  btg.gram_table(gram = [2,3,4],length=10,text = gram_analysis2)
  btg.gram_table(gram = [2,3,4],length=10,text = gram_analysis3)
  btg.gram_table(gram = [2,3,4],length=10,text = gram_analysis4)
  
  #how concern are correlated by using chi-square
  chi_df = pd.pivot_table(new_df,values = 'Score',index = ['Text'],columns = ['Concern_exp'],aggfunc = np.min)
  chi_df = chi_df.reset_index()
  chi_df = chi_df.fillna(0)
  
  
  
  
  
  
      #num=int(len(train['Label'][1]))
    # roberta model
#    roberta = MultiLabelClassificationModel('roberta','roberta-base',num_labels = num,
#                                            args = {'output_dir':'outputs_roberta/',
#                                                    'train_batch_size':8,
#                                                    'gradient_accumulation_steps':5,
#                                                    'learning_rate':0.005,
#                                                    'num_train_epochs':5,
#                                                    'max_seq_length':512,
#                                                    'overwrite_output_dir':True},
#                                            use_cuda=False)
#    roberta.train_model(train)
#    result_r,model_outputs_r,wrong_predictions_r=roberta.eval_model(test)
    
    
#    # Albert model
#    albert = MultiLabelClassificationModel('albert','albert-base-v1',num_labels = num,
#                                        args = {'output_dir':'outputs_albert/','train_batch_size':8,'gradient_accumulation_steps':5,
#                                                'learning_rate':0.0005,'num_train_epochs':5,
#                                                'max_seq_length':512,'overwrite_output_dir':True},
#                                           use_cuda=False)
#
#    albert.train_model(train)
#    result_a,model_outputs_a,wrong_predictions_a=albert.eval_model(test)
    
  # Distilbert model
#  distilbert = MultiLabelClassificationModel('distilbert','distilbert-base-uncased',num_labels = num,
#                                        args = {'output_dir':'outputs_disbert/',
#                                                'train_batch_size':8,
#                                                'gradient_accumulation_steps':5,
#                                                'learning_rate':0.0005,
#                                                'num_train_epochs':5,
#                                                'max_seq_length':512,
#                                                'overwrite_output_dir':True},
#                                               use_cuda=False)
#
#    distilbert.train_model(train)
#    result_d,model_outputs_d,wrong_prediction_d=distilbert.eval_model(test)
#    accy = {result_d_acc,model_outputs_d_acc,wrong_predictions_d_acc}
#    data = {'Roberta':result_r,'Albert':result_a,'Distilbert':result_d}
#    model_eval = pd.DataFrame(data)


