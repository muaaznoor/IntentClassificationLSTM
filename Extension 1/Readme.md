# Extension 1: Hierarchical Modeling

Similar to the strong baseline model, the extended LSTM model takes in tokenized user utterances, gets their Glove embeddings, passes them through a bidirectional LSTM and fully connected layers, and outputs the predictions of intent.

The main difference between the strong baseline model and extended model is the way that loss is calculated. Instead of measuring loss solely based on the given labels, we place those that are related to each other into the same category and give them a parent label. To avoid confusion, we call the given labels “child labels” and the created labels “parent labels”. Since we are required to predict both parent and child labels, there are some minor changes to the baseline architecture: the input to the extended LSTM model is padded with an additional <PAD> token at the end of all text samples. 

The output of the actual last token of the sentences produces the parent class prediction. The output of the <PAD> token produces the child class prediction, which is the one used to compute classification accuracy. The loss of the extended model becomes the sum of the losses pertaining to the parent class prediction and child class prediction.

Here is a diagram of the neural network architecture of our extended model:

![Picture3](https://github.com/muaaznoor/IntentClassificationLSTM/assets/97204777/7db34159-f4a8-47eb-8531-61ff8b46ba09)

We use the same hyperparameters that were used to train the strong baseline model. The extended LSTM model produces a testing accuracy of around 87% which is approximately 1.5% higher than the testing accuracy of the strong baseline.
