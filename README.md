# Question Answering in SQuAD and BioASQ

Each branch of the project has a different modification implemented. The master branch contains 
the original model.

1. [Introduction](#introduction)
2. [Changing to bio related embeddings & fine tuning in BioASQ dataset](#changing-to-bio-related-embeddings--fine-tuning-in-bioasq-dataset-change-combine)
3. [Combining RNN encoding with an CNN encoding](#combining-rnn-encoding-with-an-cnn-encoding)
4. [Adding character embeddings](#adding-character-embeddings)

## Introduction

This project has as its starting point the given model trained in the SQuAD dataset. We have implemented a series of changes with two objectives in mind. First, to improve the general performance of the model, and second, to improve the generalization of the model in the BioASQ dataset. The starting performance of the model is the following:

* SQuAD: {'EM': 50.94, 'F1': 63.92}

* BioASQ: {'EM': 12.62, 'F1': 22.79}

Therefore, more specifically, the objective of the changes implemented is to get better results in the BioASQ the SQuAD datasets. This objective was in fact achieved by using bio related embeddings and fine tuning (this change will be further explained in the next section). The results obtained by our model were the following:

* SQuAD predictions: {'EM': 37.28, 'F1': 48.92}

* BioASQ predictions: {'EM': 54.49, 'F1': 64.94}

<br/>

The following table compared the result obtained by the different changes:

<table>
  <tr>
    <td>Predictions</td>
    <td>SQuAD</td>
    <td>BioASQ</td>
  </tr>
  <tr>
    <td>Original</td>
    <td>{'EM': 50.94, 'F1': 63.92}</td>
    <td>{'EM': 12.62, 'F1': 22.79}</td>
  </tr>
  <tr>
    <td>Bio embeddings</td>
    <td>{'EM': 37.28, 'F1': 48.92}</td>
    <td>{'EM': 54.49, 'F1': 64.94}</td>
  </tr>
  <tr>
    <td>Channels: 50/50</td>
    <td>{'EM': 49.25, 'F1': 61.67}</td>
    <td></td>
  </tr>
</table>
<br/>

The changes have been divided in sections in which its implementation and results are further explained. Each of these changes will be targeted at one of the two objectives mentioned above.

## Changing to bio related embeddings & fine tuning in BioASQ dataset ([change](https://github.com/vibalcam/nlp-finalProject/tree/change_embeddings), [combine](https://github.com/vibalcam/nlp-finalProject/tree/combine_embeddings))

The first change is to first train the model with the SQUAD dataset, change the embeddings to bio related embeddings and fine tune the model on the BioASQ dataset. The bio related embeddings that we have used are medical word embeddings which can be found in this GitHub repository:

[https://github.com/basaldella/bioreddit](https://github.com/basaldella/bioreddit) 

The first issue is when to change these embeddings. Having in mind that the original model stops improving after the third epoch, we started our trial in this epoch. In order to fine tune the model in the BioASQ dataset, we divided it into train and development sets.

In the beginning, we also used different vocabularies for each distinct dataset. This approach gave us the following results:

*  SQuAD predictions: {'EM': 2.61, 'F1': 7.55}

* BioASQ predictions: {'EM': 37.87, 'F1': 46.29}

Even though we have greatly improved generalization in the BioASQ dataset, the results in SQUAD have been severely lowered. The cause could be one of the following:

* The vocabulary of the second dataset does not have many words that were in the first one and due to this reason, the predictions in the first dataset are not accurate enough.

* The second embeddings are too different, and they are not able to predict correctly sentences of the first dataset 

These results did not satisfy us. Thus, we looked for another way which did not lower as much the SQuAD performance. We started by, instead of changing the vocabulary, implementing a combined vocabulary with the samples of the SQuAD training data and the BioASQ dataset. Now we only change the embeddings, not the words of the vocabulary. This approach gave us the following results:

* SQUAD predictions: {'EM': 2.33, 'F1': 7.78}

* BioASQ predictions: {'EM': 34.88, 'F1': 43.55}

The results are very similar to the ones obtained before. Therefore, the problem remained, and this fix did not work.

The next modification we have implemented to try to fix the previous problem is using two layers of embeddings so we can have both, glove and medical embeddings.

The first step is using two embedding layers instead of only one. In order to obtain this, we have concatenated these layers and passed them to the next layer. As a result, the dimension of the input of the next layer is 600, twice the former size. We are going to still use a unique vocabulary, so we have the same words in both layers. The results with this second approach are the following:

* SQuAD predictions: {'EM': 37.28, 'F1': 48.92}

* BioASQ predictions: {'EM': 54.49, 'F1': 64.94}

We can see a significant improvement in both performances compared with our previous modifications. Even though it is true that predictions in the first dataset are worse compared to the starting point, we have achieved a model that is able to have decent predictions in both datasets. We have greatly improved the model’s performance in BioASQ and minimized the loss of performance in SQuAD.

## [Combining RNN encoding with an CNN encoding](https://github.com/vibalcam/nlp-finalProject/tree/adding_CNN)

This next change has as its objective to increase the model’s general performance. The RNNs used in the model are bidirectional, meaning that they look at the sentences from the beginning to the end and from the end to the beginning. The hypothesis behind this change is to include a CNN that acts as an n-gram. Thus, the CNN will look n words around each word and encode them. This will give more information to the model about the context. Instead of summing the RNN and the CNN hidden layers or concatenating them, we opted to use a linear layer as a combination layer. By using a linear layer, the model can give importance to the RNN and the CNN depending on which performs best. This is a better way to combine both hidden layers since it lets the model weight their contribution, instead of having it hardcoded. 

The RNN used was the same as the original model used. For the CNN we used four blocks composed of two one-dimensional convolutions with ReLUs after each convolution. The convolutions used for these blocks had increasing dilation (dilation_n = 2 ^ n) and used a kernel size of three. After these four blocks, a one-dimensional convolution with kernel size of one was used as a classifier. The output of the classifier has a number of channels equal to the hidden dimension. Since the RNNs used are bidirectional, their output have twice the hidden dimensions. Therefore, the classifier that combines the outputs of the RNN and the CNN, has as its input dimension three times the hidden dimension, and outputs twice the hidden dimension. The reason for this output is to leave the rest of the model unchanged.

Two variations of this model were tried. Both of which were trained and evaluated on the SQuAD dataset. The first one used channels of dimension 50 for all the intermediate convolutional layers (all except the input of the first layer and output of the classification layer). The second one used channels of dimension 200 for the passage and 100 for the question. The following results were obtained:

<table>
  <tr>
    <td>SQuAD predictions</td>
    <td>EM</td>
    <td>F1</td>
  </tr>
  <tr>
    <td>Original</td>
    <td>50.94</td>
    <td>63.92</td>
  </tr>
  <tr>
    <td>Channels: 50/50</td>
    <td>49.25</td>
    <td>61.67</td>
  </tr>
  <tr>
    <td>Channels: 200/100</td>
    <td>44.03</td>
    <td>56.29</td>
  </tr>
</table>
<br/>

The model results are very similar to the ones obtained by the original model. This is probably because the CNN does not perform well, and the classifier gives it a small weight. This would explain the slightly lower results obtained. This is also suggested by the fact that the results get worse as we increase the number of channels. Increasing the number of channels increases the number of parameters of the model, which means that the model’s complexity increases. This makes it harder for the classifier to correctly do its job.

The results suggest that this change does not improve the model. Something else that could be tried would be to increase the kernel size or use another non-linear layer such as tanh. However, given the results, we opted to abandon this route and look for another way to improve the general performance of the model.

## [Adding character embeddings](https://github.com/vibalcam/nlp-finalProject/tree/letter_embeddings)

After viewing the results obtained by our last change, we opted to abandon it and try something new. The idea we came up with was to improve the model’s performance with words that it has not seen during training. By doing so, we could theoretically improve the general performance of the model, both in the SQuAD and BioASQ datasets. 

With this objective in mind, we have added a character embeddings layer. This layer will allow the model to have some information about the word even if it has not seen it before. This layer works as follows. It uses a torch Embeddings which will train with the model. This layer takes the sentence as input, obtains the embeddings for each character, and averages the character embeddings of each word. The output of this layer is concatenated with the embeddings obtained by the original model (passage and question embeddings). The size of the embeddings used was 25.

However, this change had a very serious performance issue: it took an average of 30 seconds per batch in a GPU (NVIDIA GeForce MX150). The main performance problem was the for-loops used to turn word embeddings into letter embeddings since some words were padding or unknown.

We then tried to improve its performance by using torch as much as possible. We used the list of passages directly and got rid of one of the for loops. We implemented batching by using a max word length of 30 and a padding token. To stop this from negatively affecting shorter words, the padding embedding was hardcoded to 0 and the mean was calculated only using the non-padding letters. Thanks to this change and by using torch’s broadcasting. We were able to greatly increase its performance to 4 seconds per batch. 

Despite the performance increase, it was still not good enough and, due to this being the last change implemented and not having a more powerful GPU, we ran out of time and were not able to test its effect.

