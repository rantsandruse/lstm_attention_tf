from lib import features_word2vec, model_lstm_tf
import pandas as pd
import os


# The next steps:
# 1. Check

# Will ingest/clean data and save the following:
# 1. cleaned text translated to array of word indices: imdb_indices.pickle
# 2. word2vec model, where the indices/word vecs are stored:  300features_40minwords_10context
# 3. word embeddings: this is the index to wordvec mapping derived from 2.

# ingestion clean data
# create word embedding
# create word indices that can be mapped to word embedding
labeled_data_path = "./data/labeledTrainData.tsv"
unlabeled_data_path = "./data/unlabeledTrainData.tsv"
model_path = "./model/300features_40minwords_10context"
embedding_path = "./model/embedding_weights.pkl"
text2indices_path = "./model/imdb_indices.pickle"



def data_prep():
    # Read data
    # Use the kaggle Bag of words vs Bag of popcorn data:
    # https://www.kaggle.com/c/word2vec-nlp-tutorial/data

    data = pd.read_csv(labeled_data_path, header=0,
                       delimiter="\t", quoting=3, encoding="utf-8")

    data2 = pd.read_csv(unlabeled_data_path, header=0,
                        delimiter="\t", quoting=3, encoding="utf-8")

    # data2 and data are combined to train word2vec model
    data2 = data.append(data2)

    model = features_word2vec.get_word2vec_model(data2, "review", num_features=300, downsampling=1e-3, model_path=model_path)
    embedding_weights = features_word2vec.create_embedding_weights(model)
    features = features_word2vec.get_indices_word2vec(data, "review", model, maxLength=500,
                                                      writeIndexFileName="./model/imdb_indices.pickle", padLeft=True)
    return model, embedding_weights, features

if __name__ == '__main__':


    # Run data prep routine if some files are not found
    if not(os.path.isfile(model_path) and os.path.isfile(embedding_path) and os.path.isfile(text2indices_path)):
        data_prep()

    #model1 = model_lstm_tf.LstmTFModel(useAttention=True, restore = False)
    #model1.train_epochs(1)
    #model1.test()


    model2 = model_lstm_tf.LstmTFModel(useAttention=True, restore = True)
    model2.train_epochs(5)
    model2.test()

    model2.plot_attention()






