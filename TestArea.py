import pickle
import json
import gensim
from gensim.test.utils import datapath
from ModelMaker import process_body

dictionary = None
data = None

with open('Raw Data/dictionary1000','rb') as fp:
    dictionary = pickle.load(fp)

fname = datapath('C:/Users/User/Desktop/Love Advice Bot/Raw Data/ldaModel1000')
lda_model = gensim.models.LdaModel.load(fname)

with open('Raw Data/data1000.json') as fp:
    data = json.load(fp)

print(data['submissions'][0]['body'])

processed_submission = process_body( data['submissions'][150]['body'], data['submissions'][150]['edited'])

bow = dictionary.doc2bow(processed_submission)
topic_prediction = lda_model.get_document_topics(bow)

print(topic_prediction)