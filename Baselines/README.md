# Baselines and Evaluation

## I. Evaluation Script

For this project, we elect to use total classification accuracy as our evaluation metric. Because we are attempting to implement models that correctly classify input sequences into multiple intent classification labels, we require a metric that reports our overall performance toward this goal.

This metric allows us to measure the proportion of inputs which our models classify correctly across all classes. It is calculated using the simple formula below:
Total Classification Accuracy =(\sum_(i=1)^n\ 1{y\ \widehat_i\ \ =\ y_i})/n

https://latex.codecogs.com/svg.image?=(%5Csum_(i=1)%5En%5C;1%7By%5C;%5Cwidehat_i%5C;%5C=%5Cy_i%7D)/n

This metric has been successfully used in a wide variety of multiclass classification studies, including the below study linked in the footnotes[1] which performs multiclass classification of intent using LSTM architectures.

A description of this accuracy metric for multiclass classification can be found in this Wikipedia article.

## II. Simple Baseline: KNN

We attempt to construct two such simple baselines: the majority class baseline and a KNN classifier baseline.

The majority class baseline simply predicts the label that shows up most frequently in the training set for all examples. In the training set, the majority class merely made up 7.035% of the dataset. Using this majority prediction, we have a training accuracy of 7.035% and testing accuracy of 7.028%, which are incredibly low but consistent hence underfit. This would be far too trivial to beat.

As a result, we tried to explore a better but still naive baseline by running a Nearest Neighbors Classifier with k=1 that uses Glove embeddings to compute the embedding of each word in an example text, then averages them to get a sentence level embedding for the example. We do this for all examples in both train and test sets. Then, for each testing example, we compute the cosine similarity of its sentence level embedding with the embedding of each of the training examples. The label of this testing example is then taken to be the same label as the training example with the highest cosine similarity. This baseline produced a testing accuracy of 74.65%, which is somewhat more realistic as a simple baseline to beat.

## III. Strong Baseline: LSTM 

The strong baseline model takes in tokenized user utterances, gets their Glove embeddings, passes them through a bidirectional LSTM and fully connected layers, and outputs the predictions of intent.

We have the following LSTM equations:

i_t=\sigma(W_ii\ x_t+b_ii+W_hi\ \begin h_(t-1)+b_hi)〗_
f_t=\sigma(W_if\ x_t+b_if+W_hf\ \begin h_(t-1)+b_hf)〗_
o_t=\sigma(W_io\ x_t+b_io+W_ho\ \begin h_(t-1)+b_ho)〗_
c_t=f_t\odot\begin c_(t-1)+i_t\odot g_t〗_
h_t=o_t\odot\begin\begin tanh(c〗_t)〗_
where we have:

h_t: hidden state at time t
c_t: cell state at time t
x_t: input at time t
h_{t-1}: hidden state at time t-1
i_t: input gate
f_t: forget fate
g_t: cell gate
o_t: output gate
\sigma(...): sigmoid activation function
\odot: Hadamard product

We use a batch size of 128, a learning rate of 0.01, and trained for 10 epochs. The strong baseline LSTM model produces a testing accuracy of around 85.5%, which is already decent but still gives us some room for improvement by doing extensions.
