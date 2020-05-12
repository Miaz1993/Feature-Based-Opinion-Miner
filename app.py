#%% Now Support Tensorflow 2.0 and Keras 2.3.1 @22DEC19

#%% Suppress warnings
#! /usr/bin/env python
import sys, os, warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#%% Import packages
from flask import Flask, render_template, flash, request
from wtforms import Form, validators, StringField
from Prediction import generate_label_df, generate_label

#%% Initial Settings

# App config
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
# Parameters
# embed_size = 300  # how big is each word vector
# max_features = 20000  # how many unique words to use (i.e num rows in embedding vector)
# maxlen = 100
# model_final = 'DL-model_300_all.h5'


#%% Main part

# Get User Input (URL)
class InputForm(Form):
    human_finding = StringField(validators=[validators.required()])


@app.route("/", methods=['GET', 'POST'])
def index():
    # get input
    form = InputForm(request.form)
    print(form.errors)
    pred = []
    pred_df_html = []
    if request.method == 'POST':
        finding = request.form['human_finding']
        if form.validate():
            flash('Generating model predictions...')

            # Acquire text from URL
            # news_text = get_text_from_url(finding)
            # # # Remove Stop words
            # # news_text = remove_stop_words(news_text)
            # # Embedding the text
            # news_text_embedding = text_embedding(news_text, max_features, maxlen)
            # Make the prediction
            pred_df,pred = generate_label(finding)
            pred_df_html = pred_df.to_html(index=False, col_space=200).\
                replace('<tr>' , '<tr style="text-align: center;">').\
                replace('<th style="' , '<th style="text-align: center;')

        else:
            flash('Error: text can not be empty.')

    return render_template('model_demo.html', form=form, pred=pred, tables=[pred_df_html])


#%% Main function
if __name__ == "__main__":
    app.run(threaded = False) # Set to false to prevent multi-threading