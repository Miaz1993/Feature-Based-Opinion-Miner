import pandas as pd

#combine nouns dictionary
#word2vec, fasttext, glove
#final nouns dictionary

#read dictionary
nouns = set(pd.read_csv('dictionary.csv')['keywords'].values)
#about 300 keywords
# nouns = {
#  'bonus',
#  'bounty',
#  'budget',
#  'buiness',
#  'busines',
#  'career',
#  'caree',
#  'careeer',
# ...
#  'communicated',
#  'communicates',
#  'communicating',
# ...
#   'quantity',
#   'objective',
#   'objectives',
#   'merit'}
 
#negative word dictionary
negative_word = {'not',"cant","wont","shouldnt",'dont','couldnt','didnt',
                 'doesnt','no','isnt','without'}

#sentiment dictionary
# from paper feature_based opinion mining
def get_sentiment_dict(file_name):
  adj_sentiment_1 = pd.read_csv(file_name)
  adj_sentiment_1 = adj_sentiment_1.groupby(['SynsetTerms']).mean().reset_index()
  adj_sentiment_1= adj_sentiment_1[['SynsetTerms','NegScore','PosScore']]
  adj_sentiment_1['Score'] = adj_sentiment_1['PosScore'] - adj_sentiment_1['NegScore']

  sentiment_dict={}
  for index, row in adj_sentiment_1.iterrows():
      sentiment_dict[row[0]] = row[3] 
  return sentiment_dict

