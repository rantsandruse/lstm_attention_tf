This is an extension of the previous project (lstm_word2vec). Now we have a tensorflow implementation of LSTM for the purpose of sentiment analysis, with the option of adding an attention layer. 

In this project, I reused the I/O function and data prep functions from lstm_word2vec.  

The major improvements are: 
1. Migration of code from keras to TF. 
2. Added an optional attention layer plus visualization. 
3. Added a batch generator class that can be easily used for different datasets.  

The main_lstm.ipynb show cases how to leverage existing functions to train your own LSTM + attention layer model. 

Addition of the attention layer appears to marginally improve the results of the testing set. But further testing is needed to establish the significance of the improvement... My intuition is that the for this particular data set, the marginally benefit of adding an additional attention layer is almost negligble. 


