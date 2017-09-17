from batch_generator import BatchGenerator
import tensorflow as tf
from lib import features_word2vec
import data_split
import pandas as pd
import datetime
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#word2vecmodel, embedding_weights, word_features, data = data_prep()
# To do tomorrow:
# 1. rename params for the alpha part
# 2. find out

class LstmTFModel:

    # initiate everything
    # embedding weights: map index to word vecs
    # word_features: map

    def __init__(self, useAttention = True, restore = False):

        self.useAttention = useAttention
        tf.reset_default_graph()
        self.session = tf.Session()
        self.restore = restore

        #if restore:
        #   self.restore_model()
        self.initialize_params()
        self.initialize_filepaths()
        self.initialize_inputs()
        self.initialize_train_test()
        self.initialize_model()
        self.initialize_tboard()


    # initialize all hyperparameters
    # including batch size, network size etc.
    def initialize_params(self):
        self.batchSize = 128
        self.lstmUnits = 64
        self.numClasses = 2
        self.maxSeqLength = 500
        self.numDimensions = 300
        self.dropOutRate = 0.20
        self.attentionSize = 128


    def initialize_filepaths(self):
        self.word2vecmodel_path = "./model/300features_40minwords_10context"
        self.embedding_path = "./model/embedding_weights.pkl"
        self.text2indices_path = "./model/imdb_indices.pickle"
        self.labeled_data_path = "./data/labeledTrainData.tsv"
        self.unlabeled_data_path = "./data/unlabeledTrainData.tsv"
        self.lstm_model_path = "./model/pretrained_lstm_tf.model"
        self.attention_map_path = "./figures/attention_map.png"


    # initialize inputs
    # word2vec model
    # word to indices vectors
    # word embeddings
    def initialize_inputs(self):
        # Read data
        # Use the kaggle Bag of words vs Bag of popcorn data:
        # https://www.kaggle.com/c/word2vec-nlp-tutorial/data
        data_labeled = pd.read_csv(self.labeled_data_path, header=0, delimiter="\t", quoting=3, encoding="utf-8")
        data_unlabeled = pd.read_csv(self.unlabeled_data_path, header=0, delimiter="\t", quoting=3, encoding="utf-8")
        # data2 and data are combined to train word2vec model
        data_combined = data_labeled.append(data_unlabeled)

        # Load or create(if not exists) word2vec model
        self.word2vecmodel = features_word2vec.get_word2vec_model(data_combined, "review", num_features=300, downsampling=1e-3,
                                                     model_path=self.word2vecmodel_path)
        # Create word embeddings, which is essentially a dictionary that maps indices to features
        self.embedding_weights = features_word2vec.create_embedding_weights(self.word2vecmodel, writeEmbeddingFileName=self.embedding_path)


        # Map words to indices
        self.X = features_word2vec.get_indices_word2vec(data_labeled, "review", self.word2vecmodel, maxLength=500,
                                                               writeIndexFileName=self.text2indices_path, padLeft=True)
        self.y = data_labeled["sentiment"]

        # convert types
        self.embedding_weights = self.embedding_weights.astype("float32")
        self.X = self.X.astype("int32")

    # Split train and test
    def initialize_train_test(self):
        self.X_train, self.y_train, self.X_test, self.y_test = data_split.train_test_split_shuffle(self.y, self.X, test_size=0.1)
        self.myBatchGenerator = BatchGenerator(self.X_train, self.y_train, self.X_test, self.y_test, self.batchSize)

    # http://aclweb.org/anthology/E17-2091
    # https://arxiv.org/pdf/1409.0473.pdf
    #
    def addAttentionToModel(self, hidden_layer):

        v_att = tf.tanh(tf.matmul(tf.reshape(hidden_layer, [-1, self.lstmUnits]), self.w_att) \
                    + tf.reshape(self.b_att, [1, -1]))
        betas = tf.matmul(v_att, tf.reshape(self.u_att, [-1, 1]))

        exp_betas = tf.reshape(tf.exp(betas), [-1, self.maxSeqLength])
        alphas = exp_betas / tf.reshape(tf.reduce_sum(exp_betas, 1), [-1, 1])

        output = tf.reduce_sum(hidden_layer * tf.reshape(alphas,
                                                         [-1, self.maxSeqLength, 1]), 1)

        return output, alphas

    # input a sentence
    # calculate alpha
    # visualize alpha.
    #def visualize_attention(self, sentence):
        # alphas_test = self.sess.run([y_hat, alphas], {batch_ph: batchtest, target_ph: batchlabel})
        # for word, coef in zip(sentence.split()[:37], alphas_test[0, :37] * 1000 / 1.7):
        #     print "\colorbox{yellow!%d}{%s}" % (int(coef), word)
        # print("yes")

    # initialize model weights, placeholders etc.
    # And model cell itself
    def initialize_model(self):

       if not self.restore:
            self.labels = tf.placeholder(tf.float32, [self.batchSize, self.numClasses], name = "labels")
            self.input_data = tf.placeholder(tf.int32, [self.batchSize, self.maxSeqLength], name = "input_data" )

            self.data = tf.Variable(tf.zeros([self.batchSize, self.maxSeqLength, self.numDimensions]), dtype=tf.float32)
            self.data = tf.nn.embedding_lookup(self.embedding_weights, self.input_data, name = "data")

            self.weight = tf.Variable(tf.random_normal([self.lstmUnits, self.numClasses], stddev=0.1), \
                                  name = "weight")
            self.bias = tf.Variable(tf.random_normal([self.numClasses], stddev=0.1), \
                                name = "bias" )

            lstmCell = tf.contrib.rnn.BasicLSTMCell(self.lstmUnits)
            self.lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=1 - self.dropOutRate)
            self.output, _ = tf.nn.dynamic_rnn(self.lstmCell, self.data, dtype=tf.float32)


            if self.useAttention:
                self.w_att = tf.Variable(tf.random_normal([self.lstmUnits, self.attentionSize], stddev=0.1), \
                                         name="w_att")
                self.b_att = tf.Variable(tf.random_normal([self.attentionSize], stddev=0.1), \
                                     name="b_att")
                self.u_att = tf.Variable(tf.random_normal([self.attentionSize], stddev=0.1), \
                                     name="u_att")
                self.last, self.alphas = self.addAttentionToModel(self.output)

            else:
                self.output = tf.transpose(self.output, [1, 0, 2])
                self.last = tf.gather(self.output, int(self.output.get_shape()[0]) - 1)

            print(self.last.get_shape())

            self.prediction = (tf.matmul(self.last, self.weight) + self.bias)
            self.correctPred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correctPred, tf.float32))
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.labels))
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

            tf.add_to_collection('output', self.output)
            tf.add_to_collection('last', self.last)
            tf.add_to_collection('prediction', self.prediction)
            tf.add_to_collection('correctPred', self.correctPred)
            tf.add_to_collection('accuracy', self.accuracy)
            tf.add_to_collection('loss', self.loss)
            tf.add_to_collection('optimizer', self.optimizer)

            if self.useAttention:
                tf.add_to_collection('alphas', self.alphas)


       else:
            self.saver = tf.train.import_meta_graph('./model/pretrained_lstm_tf.model-0.meta')
            self.saver.restore(self.session, tf.train.latest_checkpoint('./model'))

            graph = tf.get_default_graph()

            self.weight = graph.get_tensor_by_name("weight:0")
            self.bias = graph.get_tensor_by_name("bias:0")
            # self.w_att = graph.get_tensor_by_name("w_att:0")
            # self.b_att = graph.get_tensor_by_name("b_att:0")
            # self.u_att = graph.get_tensor_by_name("u_att:0")

            self.labels = graph.get_tensor_by_name('labels:0')
            self.input_data = graph.get_tensor_by_name('input_data:0')
            self.data = graph.get_tensor_by_name('data:0')

            #self.lstmCell = tf.get_collection('lstmCell')
            self.output = tf.get_collection('output')[0]
            if self.useAttention:
                self.w_att = tf.Variable(tf.random_normal([self.lstmUnits, self.attentionSize], stddev=0.1), \
                                     name="w_att")
                self.b_att = tf.Variable(tf.random_normal([self.attentionSize], stddev=0.1), \
                                name="b_att")
                self.u_att = tf.Variable(tf.random_normal([self.attentionSize], stddev=0.1), \
                                name="u_att")
                self.alphas = tf.get_collection('alphas')[0]

            self.last = tf.get_collection('last')[0]
            self.prediction = tf.get_collection('prediction')[0]
            self.correctPred = tf.get_collection('correctPred')[0]
            self.accuracy = tf.get_collection('accuracy')[0]
            self.loss = tf.get_collection('loss')[0]
            self.optimizer = tf.get_collection('optimizer')[0]


            # self.weight = tf.get_variable("softmax_w", [self.lstmUnits, self.numClasses])
            # self.bias = tf.get_variable("softmax_b", [self.numClasses])


    # initialize tensor board for monitoring
    def initialize_tboard(self):
        tf.summary.scalar('Loss', self.loss)
        tf.summary.scalar('Accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()
        self.logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        self.writer = tf.summary.FileWriter(self.logdir, self.session.graph)
        if not self.restore:
            self.saver = tf.train.Saver()

    def train_single_epoch(self, epoch_num = 0):
        i = 0
        while True:
            # Next Batch of reviews
            nextBatch, nextBatchLabels = self.myBatchGenerator.nextTrainBatch()
            if len(nextBatch) * (i+1) > len(self.X_train): break

            self.session.run(self.optimizer, {self.input_data: nextBatch, self.labels: nextBatchLabels})

            # Write summary to Tensorboard
            if (i % 50 == 0):
                summary, acc, cost = self.session.run([ self.merged, self.accuracy, self.loss], {self.input_data: nextBatch, self.labels: nextBatchLabels})
                print "Iter " + str(i) + ", Minibatch Loss= " + "{:.6f}".format(cost) + \
                      ", Training Accuracy= " + "{:.5f}".format(acc)

            i += 1

    def train_epochs(self, n_epochs):
        if not self.restore:
            tf.global_variables_initializer().run(session=self.session)

        num = 0
        while num < n_epochs:
            print("Epoch " + str(num) + ":\n")
            self.train_single_epoch(num)
            self.test()

            if num % 5 == 0:
                self.save_model(num)
                print("saved to %s" % self.lstm_model_path)

            num += 1

        self.writer.flush()
        self.writer.close()
        print('training finished.')

    def test(self):
        i = correct = total = 0

        while True:
            nextBatch, nextBatchLabels = self.myBatchGenerator.nextTestBatch()
            if len(nextBatch) * (i + 1) > len(self.X_test): break

            acc = self.session.run(self.accuracy, {self.input_data: nextBatch, self.labels: nextBatchLabels})
            correct += acc
            total += len(nextBatch)

            i += 1

        total_accuracy = correct/(i-1)
        print("Testing accuracy = " + "{:.5f}".format(total_accuracy))

    # Shows attention for each word
    def plot_attention(self, ind = 0 ):
        nextBatch, nextBatchLabels = self.myBatchGenerator.nextTestBatch()

        my_sentence = nextBatch[ind]
        my_alphas = self.session.run(self.alphas, {self.input_data: nextBatch, self.labels:nextBatchLabels})
        my_words = features_word2vec.indices_to_words(my_sentence, self.word2vecmodel, maxLength=self.maxSeqLength)
        my_indices = xrange(self.maxSeqLength)
        my_dummies = [""] * self.maxSeqLength

        # find the first none-zero element
        for j in xrange(len(my_sentence)):
            if my_sentence[j] != 0: break

        words_with_index = ["{:03}_{}".format(i, w) for i, w in zip(my_indices, my_words)]
        words_for_annot = np.array([[w] for w in my_words ])

        # get the attention of 35 words
        # customizable at this point.
        wordmap = pd.DataFrame({"words": words_with_index[j:j+35], "alphas": my_alphas[ind][j:j+35], "dummy": my_dummies[j:j+35]})
        mymap = wordmap.pivot("words", "dummy", "alphas")



        print(words_for_annot.shape)
        my_plot = sns.heatmap(mymap, annot=words_for_annot[j:j+35], yticklabels=my_dummies, fmt="")

        # This sets the yticks "upright" with 0, as opposed to sideways with 90.
        plt.yticks(rotation=0)
        fig = my_plot.get_figure()
        fig.savefig(self.attention_map_path)

    def save_model(self, step_num ):
        self.saver.save(self.session,
                     self.lstm_model_path, global_step=step_num)

