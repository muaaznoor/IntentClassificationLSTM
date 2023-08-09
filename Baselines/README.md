# Baselines and Evaluation

## I. Evaluation Script

For this project, we elect to use total classification accuracy as our evaluation metric. Because we are attempting to implement models that correctly classify input sequences into multiple intent classification labels, we require a metric that reports our overall performance toward this goal.

This metric allows us to measure the proportion of inputs which our models classify correctly across all classes. It is calculated using the simple formula below:

**Total Classification Accuracy** =<img src="http://www.sciweavers.org/tex2img.php?eq=%3D%5Cfrac%7B%5Csum_%7Bi%3D1%7D%5En%201%5C%7B%5Chat%7By_i%7D%20%5C%20%3D%5C%20y_i%5C%7D%7D%7Bn%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="=\frac{\sum_{i=1}^n 1\{\hat{y_i} \ =\ y_i\}}{n}" width="157" height="47" />

This metric has been successfully used in a wide variety of multiclass classification studies, including the below study linked in the footnotes[1] which performs multiclass classification of intent using LSTM architectures.

A description of this accuracy metric for multiclass classification can be found in this Wikipedia article.

## II. Simple Baseline: KNN

We attempt to construct two such simple baselines: the majority class baseline and a KNN classifier baseline.

The majority class baseline simply predicts the label that shows up most frequently in the training set for all examples. In the training set, the majority class merely made up 7.035% of the dataset. Using this majority prediction, we have a training accuracy of 7.035% and testing accuracy of 7.028%, which are incredibly low but consistent hence underfit. This would be far too trivial to beat.

As a result, we tried to explore a better but still naive baseline by running a Nearest Neighbors Classifier with k=1 that uses Glove embeddings to compute the embedding of each word in an example text, then averages them to get a sentence level embedding for the example. We do this for all examples in both train and test sets. Then, for each testing example, we compute the cosine similarity of its sentence level embedding with the embedding of each of the training examples. The label of this testing example is then taken to be the same label as the training example with the highest cosine similarity. This baseline produced a testing accuracy of 74.65%, which is somewhat more realistic as a simple baseline to beat.

## III. Strong Baseline: LSTM 

The strong baseline model takes in tokenized user utterances, gets their Glove embeddings, passes them through a bidirectional LSTM and fully connected layers, and outputs the predictions of intent.

We have the following LSTM equations:

<img src="http://www.sciweavers.org/tex2img.php?eq=i_t%3D%5Csigma%28W_%7Bii%7D%5C%20x_t%2Bb_%7Bii%7D%2BW_%7Bhi%7D%5C%20h_%7B%28t-1%29%7D%2Bb_%7Bhi%7D%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="i_t=\sigma(W_{ii}\ x_t+b_{ii}+W_{hi}\ h_{(t-1)}+b_{hi})" width="292" height="21" />

<img src="http://www.sciweavers.org/tex2img.php?eq=f_t%3D%5Csigma%28W_%7Bif%7D%5C%20x_t%2Bb_%7Bif%7D%2BW_%7Bhf%7D%5C%20h_%7B%28t-1%29%7D%2Bb_%7Bhf%7D%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="f_t=\sigma(W_{if}\ x_t+b_{if}+W_{hf}\ h_{(t-1)}+b_{hf})" width="303" height="21" />

<img src="http://www.sciweavers.org/tex2img.php?eq=o_t%3D%5Csigma%28W_%7Bio%7D%5C%20x_t%2Bb_%7Bio%7D%2BW_%7Bho%7D%5C%20h_%7B%28t-1%29%7D%2Bb_%7Bho%7D%29%E3%80%97_&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="o_t=\sigma(W_{io}\ x_t+b_{io}+W_{ho}\ h_{(t-1)}+b_{ho})〗_" width="314" height="21" />

<img src="http://www.sciweavers.org/tex2img.php?eq=c_t%3Df_t%5Codot%5C%20c_%7Bt-1%7D%2Bi_t%5Codot%20g_t&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="c_t=f_t\odot\ c_{t-1}+i_t\odot g_t" width="182" height="19" />

<img src="http://www.sciweavers.org/tex2img.php?eq=h_t%3Do_t%5Codot%5Ctanh%28c_t%29&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="h_t=o_t\odot\tanh(c_t)" width="149" height="18" />

where we have:

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cbegin%7Bflushleft%7D%0Ah_t%3A%20%5Ctext%7Bhidden%20state%20at%20time%20t%7D%0A%5C%5Cc_t%3A%20%5Ctext%7Bcell%20state%20at%20time%20t%7D%5C%5C%0Ax_t%3A%20%5Ctext%7Binput%20at%20time%20t%7D%5C%5C%0Ah_%7Bt-1%7D%3A%20%5Ctext%7Bhidden%20state%20at%20time%20t-1%7D%5C%5C%0Ai_t%3A%20%5Ctext%7Binput%20gate%7D%5C%5C%0Af_t%3A%20%5Ctext%7Bforget%20fate%7D%5C%5C%0Ag_t%3A%20%5Ctext%7Bcell%20gate%7D%5C%5C%0Ao_t%3A%20%5Ctext%7Boutput%20gate%7D%5C%5C%0A%5Csigma%28...%29%3A%20%5Ctext%7Bsigmoid%20activation%20function%7D%5C%5C%0A%5Codot%3A%20%5Ctext%7BHadamard%20product%7D%0A%5Cend%7Bflushleft%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\begin{flushleft}h_t: \text{hidden state at time t}\\c_t: \text{cell state at time t}\\x_t: \text{input at time t}\\h_{t-1}: \text{hidden state at time t-1}\\i_t: \text{input gate}\\f_t: \text{forget fate}\\g_t: \text{cell gate}\\o_t: \text{output gate}\\\sigma(...): \text{sigmoid activation function}\\\odot: \text{Hadamard product}\end{flushleft}" width="285" height="200" />

We use a batch size of 128, a learning rate of 0.01, and trained for 10 epochs. The strong baseline LSTM model produces a testing accuracy of around 85.5%, which is already decent but still gives us some room for improvement by doing extensions.
