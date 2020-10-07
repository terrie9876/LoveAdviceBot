import json
import gensim
import pickle
import re
from num2words import num2words
import markovify
from gensim.test.utils import datapath

data = None
dictionary = None
topic_sizes = [5,10,15,20]


def process_comment(comment):
    ans = re.sub(r"\n+", ' ', comment)  # remove newline characters
    ans = re.sub(r"\su\s", ' you ', ans)  # changing when people use u to you
    ans = ans.replace('<3', '')

    #   Editing funky punctuations
    ans = ans.translate({ord(i): '.' for i in '!?'})  # do this to help with markovify
    ans = ans.translate({ord(i): None for i in '>“”\'*:'})  # removing quotes and other noisy punctuation
    ans = ans.translate({ord(i): ' ' for i in '—/\\'})
    ans = ans.replace('...', '.')

    ans = re.sub(r"\s?\((.*?)\)\s?", "", ans)  # Remove everything that is surrounded by parenthesis
    ans = re.sub(r'edit(:)?', '', ans, flags=re.IGNORECASE)  # People add edits to comments too

    #   turning all digits into their written form
    #   normal digits
    ans = re.split(r'\s+([0-9]+)\s+', ans)
    #   Hacky-misses case of two numbers adjacent to each other
    for i in range(1, len(ans), 2):
        ans[i] = num2words(int(ans[i]))
    ans = ' '.join(ans)
    #   possible half cardinals (e.g 1st 2nd 3rd 568th)
    ans = re.split(r'\s+([0-9]+)[a-zA-Z]{2}\s+', ans)
    for i in range(1, len(ans), 2):
        ans[i] = num2words(int(ans[i]), ordinal=True)
    ans = ' '.join(ans)
    ans = ans.translate({ord(i): ' ' for i in '-'})

    return ans

for num_topics in topic_sizes:
    # load in lda_model
    fname = datapath('C:/Users/User/Desktop/Love Advice Bot/Raw Data/ldaModel' + str(num_topics))
    lda_model = gensim.models.LdaModel.load(fname)

    sorted_comments = [[] for i in range(num_topics)]

    with open('Raw Data/preData.json') as fp:
        data = json.load(fp)

    with open('Raw Data/dictionary', 'rb') as fp:
        dictionary = pickle.load(fp)

    count = 0
    for submission in data['submissions']:

        bow = dictionary.doc2bow(submission['body'])
        topic_prediction = lda_model.get_document_topics(bow, minimum_probability=.2)
        topic_prediction.sort(key=lambda xx: xx[1], reverse=True)

    #   Now we need to sort the comments into the proper categories depending on the probabilities given
        comments = submission['comments']
        num_labels = len(topic_prediction)
        if num_labels == 1:
            for comm in comments:
                sorted_comments[topic_prediction[0][0]].append(comm)
        elif num_labels == 2:
            # Either a 4-1 or a 3-2 spilt in the comments
            x = 3
            if topic_prediction[0][1] > .7:
                x = 4
            for i in range(x):
                sorted_comments[topic_prediction[0][0]].append(comments[i])
            for i in range(x, 5):
                sorted_comments[topic_prediction[1][0]].append(comments[i])
        elif num_labels == 3:
            # Either a 3-1-1 or a 2-2-1 spilt
            x = 2
            if topic_prediction[0][0] > .5:
                x = 3
            for i in range(x):
                sorted_comments[topic_prediction[0][0]].append(comments[i])
            for i in range(x, 4):
                sorted_comments[topic_prediction[1][0]].append(comments[i])
            sorted_comments[topic_prediction[2][0]].append(comments[4])
        else:
            # 2-1-1-1 split
            sorted_comments[topic_prediction[0][0]].append(comments[0])
            sorted_comments[topic_prediction[0][0]].append(comments[1])
            sorted_comments[topic_prediction[1][0]].append(comments[2])
            sorted_comments[topic_prediction[2][0]].append(comments[3])
            sorted_comments[topic_prediction[3][0]].append(comments[4])

    print('\n\n\n\n********Markov GENERATORS**********\n\n\n\n')

    markov_topics = []
    # Processing all the comments of a topic then add it to markov_topics
    for t in range(len(sorted_comments)):
        for c in range(len(sorted_comments[t])):
            sorted_comments[t][c] = process_comment(sorted_comments[t][c])
        markov_topics.append(' '.join(sorted_comments[t]))

    markov_models = []

    for corpus in markov_topics:
        markov_models.append(markovify.Text(corpus))

    for x in range(len(markov_models)):
        for i in range(5):
            print(markov_models[x].make_sentence(), end='\n**********NEXT********^' + str(x) + '\n')
        # Make json files for each of the Markov Chains
        with open('Raw Data/MarkovModels/markov' + str(num_topics)+ '_' + str(x) + '.json', 'w') as fp:
            json.dump(markov_models[x].to_json(), fp)

