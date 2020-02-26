# A CNN that categorizes Math exercises
Work in progress repository that implements Multi-Class Text Classification Using CNN (Convolutional Neural Network) made for a [Deep Learning university exam](http://www.unife.it/ing/lm.infoauto/deep-learning/scheda-insegnamento-1/en) using [PyTorch](https://github.com/pytorch/pytorch), [TorchText](https://github.com/pytorch/text) and Python 3.7.
It also integrates [TensorboardX](https://github.com/lanpa/tensorboardX) a module for visualization with Google’s tensorflow’s tensorboard for PyTorch, a web server to serve visualizations of the training progress of a neural network. 
TensorboardX is used in order to visualize embedding, PR and Loss/Accuracy curves. 

I've started from [this awesome tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis) that perfectly shows how to perform sentiment analysis with PyTorch.

The achievement of this particular application of a [convolutional neural network](https://arxiv.org/abs/1408.5882) (CNN) model to Text Classification is to being able to categorize 7 different classes of Italian Math/Calculus exercises, using a small, but balanced dataset.
For example, given a 4D optimization Calculus exercise to the input, the NN should be able to categorize that exercise as a 4D optimization problem.




## Getting Started

To install PyTorch, see installation instructions on the [PyTorch website](pytorch.org).

To install TorchText, Inquirer, TorchSummary and TensorboardX:

``` bash
pip install inquirer
pip install tensorboardX
pip install torchsummary
pip install torchtext
```

 SpaCy is required to tokenize our data. To install spaCy, follow the instructions [here](https://spacy.io/usage/) making sure to install the Italian models with:

``` bash
python -m spacy download it_core_news_sm
```

I've used two different embeddings - to build the vocab and load the pre-trained word embeddings - you can download them here:
- [Human Language Technologies - CNR](http://hlt.isti.cnr.it/wordembeddings/)
- [Suggested] [Italian CoNLL17 corpus](http://vectors.nlpl.eu/repository/) (filtering by language) 

You should extract one of them to **vector_cache** folder and load it from **dataset.py**.
For example:
```python
vectors = vocab.Vectors(name='model.txt', cache='vector_cache/word2vec_CoNLL17')
```

I'd anyway suggest to use Word2Vec models as I've found them easier to integrate with libraries such [nlpaug - Data Augmentation for NLP](https://github.com/makcedward/nlpaug) (Not used atm)

After have performed any TensorboardX related operation remember to run 
``` bash
 tensorboard --logdir=tensorboard    
```

## References

- http://anie.me/On-Torchtext/
- https://github.com/lanpa/tensorboardX
- https://radimrehurek.com/gensim/models/word2vec.html#module-gensim.models.word2vec
- https://stats.stackexchange.com/questions/164876/tradeoff-batch-size-vs-number-of-iterations-to-train-a-neural-network
- https://magmax.org/python-inquirer/examples.html
- https://pytorch.org/docs/stable/tensorboard.html?highlight=tensorboard
- http://www.erogol.com/use-tensorboard-pytorch/
- https://github.com/sksq96/pytorch-summary

