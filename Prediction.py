import numpy as np
import pandas as pd
from nlp_predict import model1, model2, model3, model4, model5
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# from tensorflow.keras.models import load_model
# from keras import backend as K

# fix random seed for reproducibility
np.random.seed(2020)

def preprocessing(text):
    text = re.sub('[^A-Za-z0-9\s]+', '', text).lower()
    text = word_tokenize(text)
    lemma = WordNetLemmatizer()
    text = [lemma.lemmatize(te) for te in text]
    text = ' '.join(text)
    return text

def generate_label_df(y_test_hat1,y_test_hat2,y_test_hat3,y_test_hat4,y_test_hat5):
    df = pd.DataFrame({'teamwork&functions_negative': y_test_hat1[:, 0],
                       'teamwork&functions_positive': y_test_hat1[:, 1],
                       'communication&understanding_negative':y_test_hat1[:, 2],
                       'communication&understanding_positive':y_test_hat1[:, 3],
                       'effectiveness_negative':y_test_hat1[:, 4],
                       'effectiveness_positive': y_test_hat1[:, 5],
                       'collaboration_negative':y_test_hat1[:, 6],
                       'collaboration_positive':y_test_hat1[:, 7],
                       'mbo&system_negative':y_test_hat1[:, 8],
                       'mbo&system_positive':y_test_hat1[:, 9],
                       'goal&targeting_negative': y_test_hat2[:,0],
                       'goal&targeting_positive':y_test_hat2[:,1],
                       'personal development_negative':y_test_hat2[:,2],
                       'personal development_positive': y_test_hat2[:, 3],
                       'promotion&rewards_positive':y_test_hat2[:, 4],
                       'promotion&rewards_negative':y_test_hat2[:, 5],
                       'compensation_negative':y_test_hat2[:, 6],
                       'compensation_positive':y_test_hat2[:, 7],
                       'leadership&management_negative':y_test_hat3[:, 0],
                       'leadership&management_positive':y_test_hat3[:, 1],
                       'training&coaching_negative':y_test_hat3[:, 2],
                       'training&coaching_positive':y_test_hat3[:, 3],
                       'expertise&talent_negative':y_test_hat3[:, 4],
                       'expertise&talent_positive':y_test_hat3[:, 5],
                       'support&resource_negative':y_test_hat3[:, 6],
                       'support&resource_positive':y_test_hat3[:, 7],
                       'change&transition_negative':y_test_hat3[:, 8],
                       'change&transition_positive':y_test_hat3[:, 9],
                       'product&project_positive':y_test_hat4[:, 0],
                       'product&project_negative':y_test_hat4[:, 1],
                       'budget&cost_negative':y_test_hat4[:, 2],
                       'budget&cost_positive':y_test_hat4[:, 3],
                       'retail&supply_negative':y_test_hat4[:, 4],
                       'retail&supply_positive':y_test_hat4[:, 5],
                       'innovation&creativity_negative':y_test_hat4[:, 6],
                       'innovation&creativity_positive':y_test_hat4[:, 7],
                       'timeline_negative':y_test_hat5[:, 0],
                       'timeline_positive': y_test_hat5[:, 2],
                       'culture&environment_negative':y_test_hat5[:, 1],
                       'culture&environment_positive':y_test_hat5[:, 3],
                       'diversity_negative':y_test_hat5[:, 4],
                       'diversity_positive':y_test_hat5[:, 5]})
    new_df = pd.DataFrame()
    for i in range(1,42,2):
        cur = df.iloc[:,i-1:i+1].sort_values(by = 0,axis = 1).iloc[:,-1]
        new_df = pd.concat([new_df,cur],axis = 1)
        new_df = new_df.sort_values(by = 0,axis = 1)
        final_df = new_df
        final_df = final_df[final_df>0.52].dropna(axis = 1)
        final = final_df.columns.values

    return final_df, final

def generate_label(X_te):
    # Make prediction
    X_te = [str(preprocessing(X_te))]
    _,y_test_hat1 = model1.predict(X_te)
    _,y_test_hat2 = model2.predict(X_te)
    _,y_test_hat3 = model3.predict(X_te)
    _,y_test_hat4 = model4.predict(X_te)
    _,y_test_hat5 = model5.predict(X_te)
    # Add clear_session after prediction
    # K.clear_session()  # So that the program can be run multi times without restart
    # Generate label for the prediction
    df, final = generate_label_df(y_test_hat1,y_test_hat2,y_test_hat3,y_test_hat4,y_test_hat5)

    return df, final

#DEMO
# res = generate_label(X_test)
# print(res)

# more training. Make it easier for employees to break away to seek out training. currently have no time.

# 'It would be nice to have a clear path of what is needed to '
#                      'get to the next step in the employees career. '
#                      'I do not know where/how to get promoted or move up in the company.'

# 'I was promoted to a different desk that required 2x times the work '
#                      'and I was not given any finical compensation to offset that effort.'
#                      ' I am being told that we are going to reevaluate it in the new year,'
#                      'but I just completed 5 months of "in season" work that I will not be getting compensated for.'

# 'I think the company has very strong leadership and goals are very clearly stated.'
#                      ' In my opinion, the goals are all achievable but tends to have a lack of '
#                      'support/equipment/resources to help obtain the goals. Personnel cut backs and '
#                      'maintenance cuts have negatively impacted our outputs as well as negatively '
#                      'impacted the morality of our workforce. I am very glad to see the company trim '
#                      'the fat and unneeded people/tools/etc. but the efforts in doing that stepped over '
#                      'the boundary of necessity and ineffective practices. The lack of resources has put'
#                      ' strain on nearly every position in our plant and there was not an incentive in conforming.'
#                      ' More work and less people/resources seems to not be able to be sustained, so we see a '
#                      'very high turnover rate because of this. From my leadership position, it also is difficult'
#                      ' to keep department morale up.'

# 'Increased communication of goals,'
#                      ' and explanation of MBO metrics. '
#                      'Some metrics to not line up with the plant goals,'
#                      ' pushing members of staff to work in silos and only '
#                      'look out for their metrics.  These metrics might affect others in the same team.'
# Reward those who achieve more than their MBO goals.

# 'Cross-functional collaboration and shared accountability is missing.'
#                       'It does not feel like marketing, finance, R&D, BD, PCMs, '
#                       'and supply chain are all working together towards shared goals '
#                       'to drive business forward.  There are natural differences in '
#                       'priorities, but without cooperation in all functions, '
#                       'projects cannot move forward.  The sense of urgency is missing in some of these functions.  '
#                       'At the leadership level, the importance of cross-functional collaboration can be '
#                       'emphasized and more direct consequences instilled when some functions routinely'
#                       ' miss critical milestones in project timelines that are imperative to delivering the year.'

# b, c = generate_label('i love brand and product')
# print(b,c)



# lst = ['Fund more on and offsite training, visiting suppliers,and other development opportunity. '
#        'As well as a better tuition reimbursement program for further education',
#        'more training. Make it easier for employees to break away to seek out training. currently have no time.']
#
# final_list = []
# for i in lst:
#     b, c = generate_label(i)
#     final_list.extend(c.values)
#
# from collections import Counter
# print(Counter(final_list))

# more training. Make it easier for employees to break away to seek out training. currently have no time.
# It would be nice to have a clear path of what is needed to get to the next step in the employees career. I do not know where/how to get promoted or move up in the company.
# Increased communication of goals, and explanation of MBO metrics. Some metrics to not line up with the plant goals, pushing members of staff to work in silos and only look out for their metrics.  These metrics might affect others in the same team.
# Cross-functional collaboration and shared accountability is missing.It does not feel like marketing, finance, R&D, BD, PCMs, and supply chain are all working together towards shared goals to drive business forward.  There are natural differences in priorities, but without cooperation in all functions, projects cannot move forward.  The sense of urgency is missing in some of these functions.  At the leadership level, the importance of cross-functional collaboration can be emphasized and more direct consequences instilled when some functions routinely miss critical milestones in project timelines that are imperative to delivering the year.'
