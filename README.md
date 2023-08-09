# Project Proposal

## I. Motivation

Classifying user intent has multitudinous applications for problems across industries. For example, in the consumer banking industry, automated chatbots take in queries from a user and return a number of suggested self-help links via the classification of their issues based on certain phrases in the input. In the airline industry, customers call to inquire about flight information and expect to be directed to the right source to execute their demands. More broadly though, virtual assistants such as Amazon Alexa are tasked with parsing language to then classify intentions in order to elicit the appropriate response. The Amazon Massive Intent Dataset provides a rich source of data to train an NLP model to accomplish this.

## II. Description of Study and Goals

Goal Statement: our goal is to build a multi-classification NLP model that ingests input from a user / customer / human and outputs a probabilistic prediction of each of the class labels that message/intention falls into.

Description of Proposed Methodology & Models: the basic architecture of the proposed model will incorporate a single layer RNN that will be built on top of an embedding layer. The RNN layer can be modified into different types, e.g. Unidirectional LSTM, Bidirectional LSTM, GRU, etc. The output of the RNN layer will be averaged out and passed into a linear layer. The linear layer will give an output that would be equal to the number of classes that are present in our dataset for the final intent classification. Applying a softmax activation function over the linear outputs will eventually give us the class probabilities. During training, we will optimize on cross-entropy loss, while during testing, we will choose the maximum probability class as the output intent of the user’s message. On the next page, we provide an illustration of our project’s roadmap and workflow.

Extension Ideas:
Instead of performing a binary classification, predictions of intention fall into one of many classes. Multiclass classification is performed with suitable feed-forward architectures to acquire the probabilities of each class.

Instead of using a single layer bidirectional LSTM, different architectures such as multiple layer LSTM or GRU may be experimented with to improve performance.

## III. Illustrative Figures

Figure 1: Project Roadmap

![Picture2](https://github.com/muaaznoor/IntentClassificationLSTM/assets/97204777/0a40bfcb-346a-4b8d-8958-b002eb1e4142)
