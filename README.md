This is an extension of the previous project (lstm_word2vec) 

In this project, I reused the I/O function and data prep functions from lstm_word2vec: 

The major improvements are: 
1. Migration code from keras to TF. 
2. Added an optional attention layer plus visualization. 
3. Added batch generator class that can be easily used for different datasets.  

The main_lstm.ipynb show cases how to leverage existing functions to train your own LSTM + attention layer model. 

Addition of the attention layer appears to marginally mprove the results of the testing set. But further testing is needed to establish the significance of the improvement... My intuition is that the for this particular data set, the marginally benefit of adding an additional attention layer is almost negligble. 


