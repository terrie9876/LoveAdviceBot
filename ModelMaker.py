#Data processing methodology procurred from: https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24

import json
import contractions
import gensim
import pickle
import re
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk

def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def process_body(body, edited=False):
    ans = re.sub(r"http\S+", "", body)  # removes any links to other reddit posts

#   want to remove summary of posts, easiest to remove end ones
    tldr_index = ans.lower().find("tl;dr")
    if tldr_index > (len(ans)//2):
        ans = ans[:tldr_index]
#   now check if the post has been edited
#   Most edit: remarks are thanking the commentors, which is not necessary for this project
    elif edited:
        edit_index = ans.lower().find("edit:")
        if edit_index > (len(ans)//2):
            ans = ans[:edit_index]
#   TODO: Look into removing tl;dr's that come at the beginning of posts
#   TODO: Same algorithm for detecting when people put Edit: beginning

#   TODO: Expand contractions
    ans = contractions.fix(ans)
#   Tokenizing the body
    resulting_process = []
#   This applies lemmatization and stemming of the tokens
    for tkn in gensim.utils.simple_preprocess(ans):
        if tkn not in gensim.parsing.preprocessing.STOPWORDS and len(tkn) > 3:
            resulting_process.append(lemmatize_stemming(tkn))

    return resulting_process


#   Need to do this check so other files can use process_body
if __name__ == '__main__':
    np.random.seed(2018)
    nltk.download('wordnet')
    temp_index = 0
    model_size = [5, 10, 15, 20]
    data = []

    with open('Raw Data/data.json') as fp:
        data = json.load(fp)

#   Preprocess all the bodies
    list_of_bodies = []
    pre_data = {'submissions': []}
    for x in range(len(data['submissions'])):
        data['submissions'][x]['body'] = process_body(data['submissions'][x]['body'], data['submissions'][x]['edited'])
        list_of_bodies.append(data['submissions'][x]['body'])

#   Creating the dictionary
    dictionary = gensim.corpora.Dictionary(list_of_bodies)

#   Getting some statistics on the generated dictionary
    total = 0
    count = 0
    for k, v in dictionary.iteritems():
        total += k
        count += 1
    print("Total number of words == " + str(total))
    print("Number of unique tokens == " + str(count))

#   making bag of words out of all the submission bodies
    bag_of_bow = [dictionary.doc2bow(sub) for sub in list_of_bodies]

    for topic_num in model_size:
        print('\n\n\n*******TRAINING FOR TOPIC SIZE =  ' + str(topic_num) + '****************\n')
        print('Starting lda training ...')
        lda_model = gensim.models.LdaModel(bag_of_bow, num_topics=topic_num, id2word=dictionary, passes=2)
        print('Finished training!')

    #   Printing the words from each topic
        for idx, topic in lda_model.print_topics(-1,10):
            print('Topic: {} \nWords: {}'.format(idx, topic))


    #   Saving all the stuff i might need to save like model, dictionary, and processed bodies
        lda_model.save('Raw Data/ldaModel'+str(topic_num))

    with open('Raw Data/dictionary', 'ab') as fp:
        pickle.dump(dictionary, fp)

    with open('Raw Data/preData.json', 'w') as fp:
        fp.seek(0)
        fp.truncate()
        json.dump(data, fp)

