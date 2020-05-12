from simpletransformers.classification import MultiLabelClassificationModel

#model 1
model1 = MultiLabelClassificationModel('distilbert', '/Users/miazhang/Desktop/Capstonefinal/outputs_disbert1/checkpoint-184-epoch-8',
                                      args={'output_dir': 'outputs_disbert1/',
                                            'train_batch_size': 8,
                                            'gradient_accumulation_steps': 5,
                                            'learning_rate': 0.0005,
                                            'num_train_epochs': 5,
                                            'max_seq_length': 512,
                                            'overwrite_output_dir': True}, use_cuda=False)
#
# model11 = MultiLabelClassificationModel('distilbert', '/Users/miazhang/Desktop/Capstonefinal/outputs_disbert1/checkpoint-161-epoch-7',
#                                       args={'output_dir': 'outputs_disbert1/',
#                                             'train_batch_size': 8,
#                                             'gradient_accumulation_steps': 5,
#                                             'learning_rate': 0.0005,
#                                             'num_train_epochs': 5,
#                                             'max_seq_length': 512,
#                                             'overwrite_output_dir': True}, use_cuda=False)
#
# model12 = MultiLabelClassificationModel('distilbert', '/Users/miazhang/Desktop/Capstonefinal/outputs_disbert1/checkpoint-138-epoch-6',
#                                       args={'output_dir': 'outputs_disbert1/',
#                                             'train_batch_size': 8,
#                                             'gradient_accumulation_steps': 5,
#                                             'learning_rate': 0.0005,
#                                             'num_train_epochs': 5,
#                                             'max_seq_length': 512,
#                                             'overwrite_output_dir': True}, use_cuda=False)

#model 2
model2 = MultiLabelClassificationModel('distilbert', '/Users/miazhang/Desktop/Capstonefinal/outputs_disbert2/checkpoint-184-epoch-8',
                                      args={'output_dir': 'outputs_disbert2/',
                                            'train_batch_size': 8,
                                            'gradient_accumulation_steps': 5,
                                            'learning_rate': 0.0005,
                                            'num_train_epochs': 5,
                                            'max_seq_length': 512,
                                            'overwrite_output_dir': True}, use_cuda=False)

# model21 = MultiLabelClassificationModel('distilbert', '/Users/miazhang/Desktop/Capstonefinal/outputs_disbert2/checkpoint-161-epoch-8',
#                                       args={'output_dir': 'outputs_disbert2/',
#                                             'train_batch_size': 8,
#                                             'gradient_accumulation_steps': 5,
#                                             'learning_rate': 0.0005,
#                                             'num_train_epochs': 5,
#                                             'max_seq_length': 512,
#                                             'overwrite_output_dir': True}, use_cuda=False)
#
# model22 = MultiLabelClassificationModel('distilbert', '/Users/miazhang/Desktop/Capstonefinal/outputs_disbert2/checkpoint-138-epoch-8',
#                                       args={'output_dir': 'outputs_disbert2/',
#                                             'train_batch_size': 8,
#                                             'gradient_accumulation_steps': 5,
#                                             'learning_rate': 0.0005,
#                                             'num_train_epochs': 5,
#                                             'max_seq_length': 512,
#                                             'overwrite_output_dir': True}, use_cuda=False)

model3 = MultiLabelClassificationModel('distilbert', '/Users/miazhang/Desktop/Capstonefinal/outputs_disbert3/checkpoint-112-epoch-8',
                                      args={'output_dir': 'outputs_disbert3/',
                                            'train_batch_size': 8,
                                            'gradient_accumulation_steps': 5,
                                            'learning_rate': 0.0005,
                                            'num_train_epochs': 5,
                                            'max_seq_length': 512,
                                            'overwrite_output_dir': True}, use_cuda=False)


model4 = MultiLabelClassificationModel('distilbert', '/Users/miazhang/Desktop/Capstonefinal/outputs_disbert4/checkpoint-75-epoch-5',
                                      args={'output_dir': 'outputs_disbert4/',
                                            'train_batch_size': 8,
                                            'gradient_accumulation_steps': 5,
                                            'learning_rate': 0.0005,
                                            'num_train_epochs': 5,
                                            'max_seq_length': 512,
                                            'overwrite_output_dir': True}, use_cuda=False)


model5 = MultiLabelClassificationModel('distilbert', '/Users/miazhang/Desktop/Capstonefinal/outputs_disbert5/checkpoint-32-epoch-8',
                                      args={'output_dir': 'outputs_disbert5/',
                                            'train_batch_size': 8,
                                            'gradient_accumulation_steps': 5,
                                            'learning_rate': 0.0005,
                                            'num_train_epochs': 5,
                                            'max_seq_length': 512,
                                            'overwrite_output_dir': True}, use_cuda=False)


# a,b = model1.predict(['no communication, they dont allow me eat hamburger'])
#
# c,d = model2.predict(['we have a very very low compensation, my manager is bad bad'])
# e,f = model3.predict(['bad senior leadership is bad because there is no training'])
# l,p = model1.predict(['KHC does not care about current employees.The concentration is '
#                       'solely on new employees, even though we wouldnt need to hire so many if there was any retention. '
#                       'So much knowledge is being lost due to lack of focus on retention.'
#                       'New employees are receiving brand new laptops, where current employees '
#                       'use old and slow computers.  More work could be done faste, Commercialization of projects with an expedited timeline that no customer actually wants or is asking for'])
#
# print(l,p)
# sub_4 = ['product&project_positive', 'product&project_negative',
#          'budget&cost_negative', 'budget&cost_positive',
#          'retail&supply_negative', 'retail&supply_positive',
#          'innovation&creativity_negative', 'innovation&creativity_positive']