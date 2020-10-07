import pickle
import json
import gensim
import markovify
from gensim.test.utils import datapath
from ModelMaker import process_body

if __name__ == '__main__':
    dictionary = None
    model_choice = 10 # Can be either 5, 10, 15, 20
    markov_models = []

    with open('Raw Data/dictionary', 'rb') as fp:
        dictionary = pickle.load(fp)

    fname = datapath('C:/Users/User/Desktop/Love Advice Bot/Raw Data/ldaModel' + str(model_choice))
    lda_model = gensim.models.LdaModel.load(fname)
    num_topics = lda_model.get_topics().shape[0]

    print('****LOADING MARKOV MODELS****\n')
    for x in range(num_topics):
        with open('Raw Data/MarkovModels/markov' + str(model_choice) + '_'+str(x)+'.json') as fp:
            model_json = json.load(fp)
            markov_models.append(markovify.Text.from_json(model_json))

    print('****PROCESSING I-O/Input.txt ****\n')

    question = ""
    with open('I-O/Input.txt', encoding="utf8") as fp:
        question = fp.read()

    question = process_body(question)

    question_as_bow = dictionary.doc2bow(question)
    topic_prediction = lda_model.get_document_topics(question_as_bow)
    list_topic = [topic[0] for topic in topic_prediction] # Gets the list of topics determined by model
    topic_weights = [topic[1] for topic in topic_prediction] # Gets weights associated with each topic
    markov_final = markovify.combine([markov_models[x] for x in list_topic], topic_weights)

    print('****GENERATING OUTPUT****\n')
    output_file = open('I-O/Output.txt', 'w')
    output_file.seek(0)
    output_file.truncate()

    # Generating two pieces of advice
    for i in range(2):
        output_file.write('****Advice '+str(i+1)+' ****\n\n')
        output = []
        num_sent = 3

        for i in range(num_sent):
            sentence = None
            while sentence is None:
                sentence = markov_final.make_short_sentence(200)
            output.append(sentence)

        final_output = ' '.join(output)
        output_file.write(final_output + '\n\n')

    print('****OUTPUT SAVED TO I-O/Output.txt****\n')
    output_file.close()
