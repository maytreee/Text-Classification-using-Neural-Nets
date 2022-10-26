# models.py

from sentiment_data import *
from evaluator import *

from collections import Counter
import os
import numpy as np
import torch
from torch import nn, optim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing.preprocessing import remove_stopword_tokens

from gensim.utils import simple_preprocess


######################################
# IMPLEMENT THE SENTIMENT CLASSIFIER #
######################################

class FeedForwardNeuralNetClassifier(nn.Module):
    """
    The Feed-Forward Neural Net sentiment classifier.
    """
    def __init__(self, n_classes, vocab_size, emb_dim, n_hidden_units):
        """
        In the __init__ function, you will define modules in FFNN.
        :param n_classes: number of classes in this classification problem
        :param vocab_size: size of vocabulary
        :param emb_dim: dimension of the embedding vectors
        :param n_hidden_units: dimension of the hidden units
        """
        super(FeedForwardNeuralNetClassifier, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
       
        # TODO: create a randomly initialized embedding matrix, and set padding_idx as 0
        self.word_embeddings = nn.EmbeddingBag(vocab_size, emb_dim)
        randon_embedding_tensor = torch.FloatTensor(np.random.rand(vocab_size, self.emb_dim))
        self.word_embeddings.wieght = nn.Parameter(randon_embedding_tensor, requires_grad=True)

        # PAD's embedding will not be trained and by default is initialized as zero

        

        # TODO: implement the FFNN architecture
        self.linear_layer_pytorch = nn.Linear(self.emb_dim, self.n_hidden_units)
        self.relu1 = nn.ReLU()
        self.linear_layer2_pytorch = nn.Linear(self.n_hidden_units, n_classes)
        self.output = nn.Softmax(dim=1)
        # when you build the FFNN model, you will need specify the embedding size using self.emb_dim, the hidden size using self.n_hidden_units,
        # and the output class size using self.n_classes        


    def forward(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> torch.Tensor:
        
        # "The forward function, which defines how FFNN should work when given a batch of inputs and their actual sent lengths (i.e., before PAD)
        # ":param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        # ":param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        # ":return the logits outputs of FFNN (i.e., the unnormalized hidden units before softmax)
         out = self.word_embeddings(batch_inputs)
         out = self.linear_layer_pytorch(out)
         out = self.relu1(out)
         out = self.linear_layer2_pytorch(out)
         out = self.output(out)
         return out

        # ": implement the forward function, which returns the logits
        # raise Exception("Not Implemented!")
         
        
    
    def batch_predict(self, batch_inputs: torch.Tensor, batch_lengths: torch.Tensor) -> List[int]:
        """
        Make predictions for a batch of inputs. This function may directly invoke `forward` (which passes the input through FFNN and returns the output logits)

        :param batch_inputs: a torch.Tensor object of size (n_examples, max_sent_length_in_this_batch), which is the *indexed* inputs
        :param batch_lengths: a torch.Tensor object of size (n_examples), which describes the actual sentence length of each example (i.e., before PAD)
        :return: a list of predicted classes for this batch of data, either 0 for negative class or 1 for positive class
        """
        # TODO: implement the prediction function, which could reuse the forward function 
        # but should return a list of predicted labels
        # raise Exception("Not Implemented!")
        out = self.forward(batch_inputs,batch_lengths)
        output = []
        for i in out:
            data = list(i)
            max_value = max(data)
            output.append(data.index(max_value))
        return output

##################################
# IMPLEMENT THE TRAINING METHODS #
##################################

def train_feedforward_neural_net(
    args,
    train_exs: List[SentimentExample], 
    dev_exs: List[SentimentExample]) -> FeedForwardNeuralNetClassifier:
    """
    Main entry point for your modifications. Trains and returns a FFNN model (whose architecture is configured based on args)

    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """

    # TODO: read in all training examples and create a vocabulary (a List-type object called `vocab`)
    vocab = [] # replace None with the correct implementation
    vocab_list = []
    with open(args.train_path) as file:
        for statement in file:
            tokenization = simple_preprocess(statement, deacc=True)
            vocab += tokenization
            # vocab += remove_stopword_tokens(tokenization, stopwords=STOPWORDS)
    word_dic = {}
    
    for word in vocab:
        if word in word_dic:
            word_dic[word] = word_dic[word] + 1
        else:
            word_dic[word] = 1
    vocab = list(set(vocab))
    for key in word_dic:
        if word_dic[key] == 1:
            vocab.remove(key)
    # for i in range(len(train_exs)):
    #     for token in train_exs[i].words:
    #         vocab_list.append(token)
    #     vocab = vocab_list
    # add PAD and UNK as the first two tokens
    # DO NOT CHANGE, PAD must go first and UNK next (as their indices have been hard-coded in several places)
    vocab = ["PAD", "UNK"] + vocab
    print("Vocab size:", len(vocab))
    # write vocab to an external file, so the vocab can be reloaded to index the test set
    with open("data/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")

    # indexing the training/dev examples
    indexing_sentiment_examples(train_exs, vocabulary=vocab, UNK_idx=1)
    indexing_sentiment_examples(dev_exs, vocabulary=vocab, UNK_idx=1)

    # TODO: create the FFNN classifier
    model = FeedForwardNeuralNetClassifier(2, len(vocab), args.emb_dim, args.n_hidden_units) # replace None with the correct implementation
    

    if args.glove_path:
        pre_vocab, embeddings =[],[]
        with open('glove.42B.300d.txt', encoding="utf-8") as data:
            full_data = data.read().strip().split('\n')
        for index in range(len(full_data)):
            word_index = full_data[index].split()[0]
            embedding_vector_index = [float(v) for v in full_data[index].split()[1:]]
            pre_vocab.append(word_index)
            embeddings.append(embedding_vector_index)
        
        final_embedding_mat = []
        for w in vocab:
            if w in pre_vocab:
                emb_index = pre_vocab.index(w)
                final_embedding_mat.append(embeddings[emb_index])
            else:
                final_embedding_mat.append([0]*300)

        model.word_embeddings.weight = nn.Parameter(torch.FloatTensor(final_embedding_mat), requires_grad = True)

    # TODO: define an Adam optimizer, using default config
    # optimizer = None # replace None with the correct implementation
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.01) 

    
    # create a batch iterator for the training data
    batch_iterator = SentimentExampleBatchIterator(train_exs, batch_size=args.batch_size, PAD_idx=0, shuffle=True)

    # training
    best_epoch = -1
    best_acc = -1
    for epoch in range(args.n_epochs):
        print("Epoch %i" % epoch)

        batch_iterator.refresh() # initiate a new iterator for this epoch

        model.train() # turn on the "training mode"
        batch_loss = 0.0
        batch_example_count = 0
        batch_data = batch_iterator.get_next_batch()
        while batch_data is not None:
            batch_inputs, batch_lengths, batch_labels = batch_data
            # TODO: clean up the gradients for this batch
            optimizer.zero_grad()

            # TODO: call the model to get the logits
            output = model(batch_inputs, batch_lengths)

            # TODO: calculate the loss (let's name it `loss`, so the follow-up lines could collect the stats)
            loss = loss_function(output, batch_labels)
            
            # record the loss and number of examples, so we could report some stats
            batch_example_count += len(batch_labels)
            batch_loss += loss.item() * len(batch_labels)

            # TODO: backpropagation (backward and step)
            loss.backward()
            optimizer.step()

            # get another batch
            batch_data = batch_iterator.get_next_batch()

        print("Avg loss: %.5f" % (batch_loss / batch_example_count))

        # evaluate on dev set
        model.eval() # turn on the "evaluation mode"
        acc, _, _, _ = evaluate(model, dev_exs, return_metrics=True)
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            print("Secure a new best accuracy %.3f in epoch %d!" % (best_acc, best_epoch))
            
            # save the current best model parameters
            print("Save the best model checkpoint as `best_model.ckpt`!")
            torch.save(model.state_dict(), "best_model.ckpt")
        print("-" * 10)

    # load back the best checkpoint on dev set
    model.load_state_dict(torch.load("best_model.ckpt"))
    
    model.eval() # switch to the evaluation mode
    return model
