
# coding: utf-8

# Written by Curtis G. Northcutt
# Contact: cgn@fb.com / curtis.northcutt@oculus.com

# # Global Transcriptions 2+ mic 2+ speakers
# 
# Some notes: The current implementation works uses transcriptions without any information directly from the audio .wav files. We use a Language Model to identify the likely-to-be-accurate transcriptions. This is fairly moronic because the transcriptions require a language model to form sentences from the speech, so we should do the cohesion at the same time as that language model. For example we have no idea when the speaker pauses speaking, forming phrases and sentences - even though this is very useful context for understanding whether a transcription makes sense (i.e. people don't usually pause right after saying "Did you... (long pause)." However, we can't gauge this with just the transcriptions because we lose the actual timing information. The main point being, our work with the language model on top of the transcriptions would be better placed if we found cohesion amoung the audio .wav files directly during transcription, and thus, only invoke a language model once. However, this is a nasty task to tackle, so instead we use current state-of-the-art transcription services, then clean up the transcriptions afterward using a separate language model.
# 
# The @ character is used when commas, dashes, and periods are used between words characters such as in 12,000 and state-of-the-art and 1.5 and take the form ['@-@', '@,@', '@.@'] in the train/test datasets
# 
# ## Dependencies
# 
# ### English word vectors: 
# https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
# ### Language Model uses RNN and loader from: 
# https://github.com/pytorch/examples/tree/master/word_language_model
# ### Transcription uses SpeechRecognition: 
# https://pypi.org/project/SpeechRecognition/


# coding: utf-8
from __future__ import print_function, absolute_import, division, unicode_literals, with_statement
# import argparse
# import time
# import math
# import os
import torch
# import torch.nn as nn
# import torch.onnx
from torch.nn.functional import softmax
import numpy as np

# import data
from egocom import rnn_model
from egocom.rnn_model import RNNModel
import pickle
import re

args_model = 'LSTM' # type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
args_emsize = 1500 # size of word embeddings
args_nhid = 1500 # number of hidden units per layer
args_nlayers = 2 # number of layers
args_dropout = 0.2 # Fraction to dropout weights during training
args_tied = False # Ties weights in LSTM (less weights if True, but less power)
args_cuda = False # Set to True if a GPU with cuda is installed.
args_seed = 1111 # seed random numbers for reproducibility
args_sequence_length = 1 # sequence length
args_save = './saved_data/model' # path where model is stored
args_batch_size = 20
eval_batch_size = 1

# Set the random seed manually for reproducibility.
torch.manual_seed(args_seed)
if torch.cuda.is_available():
    if not args_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args_cuda else "cpu")

# Set up model and data-structures for corpus.

# Load saved corpus dictionary
with open(args_save + ".dict", 'rb') as f:
    corpus_dict = pickle.load(f)
corpus_word_set = set(corpus_dict.idx2word)
ORIG_NTOKENS = len(corpus_word_set)
# Load the saved model.
model = rnn_model.RNNModel(args_model, ORIG_NTOKENS, args_emsize, args_nhid, args_nlayers, args_dropout, args_tied).to(device)
model.load_state_dict(torch.load(args_save + ".state_dict.pt"))
# after load the rnn params are not a continuous chunk of memory
# this makes them a continuous chunk, and will speed up forward pass
model.rnn.flatten_parameters()


# Create dictionary mapping word/idx to the probability that word starts a sentence.
eos_idx = torch.from_numpy(np.array([corpus_dict.word2idx['<eos>']]).reshape((1,1)).astype(int)) # creates Tensor([[0]]) which represents the <eos> starting token.
idx2probstart = softmax(model(eos_idx, model.init_hidden(eval_batch_size))[0].view(-1, ORIG_NTOKENS), dim=1)[0].detach().numpy()
word2probstart = dict(zip([corpus_dict.idx2word[z] for z in np.argsort(idx2probstart)[::-1]], [float(z) for z in np.sort(idx2probstart)[::-1]]))


def tokenize(text, dictionary, training = False):
    """Tokenizes a string containing multiple sentences. 
    Sentences are split by a period surrounded by spaces on both sides like `a . b'.
    Lines are split by '\n' chars."""
     
    orig_ntokens = len(dictionary)
    tokens = 0
    for line in text.split('\n'):
        words = line.split() + ['<eos>']
        tokens += len(words)
        if training:
            # Add words to the dictionary
            for word in words:
                dictionary.add_word(word)

    # Tokenize file content
    ids = torch.LongTensor(tokens)
    token = 0
    for line in text.split('\n'):
        words = line.split() + ['<eos>']
        for word in words:
            ids[token] = dictionary.word2idx[word]
            token += 1

    return ids, orig_ntokens

def batchify(data, bsz):
    '''Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    batch processing.'''
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i):
    '''get_batch subdivides the source data into chunks of length args.sequence_length.
    If source is equal to the example output of the batchify function, with
    a sequence_length-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.'''    
#     print(source, i, len(source))
    seq_len = min(args_sequence_length, len(source) - 1 - i)
#     print(seq_len)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
#     print(data, target)
    return data, target


def evaluate(data_source, ntokens):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
#     print(hidden, type(hidden), len(hidden), hidden[0].shape)
    result = []
#     print("data_source size", data_source.size())
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args_sequence_length):
            data, targets = get_batch(data_source, i)
#             print(hidden[0])
#             print(hidden.size)
#             print(hidden, type(hidden), len(hidden), hidden[0].shape)
#             print(data, type(data), len(data), data.shape)
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, len(corpus_dict.idx2word))
            result.append(softmax(output_flat, dim=1))
           
    return result


def replace_words_not_in_corpus_with_unk(text):
    '''Replaces words not in our trained corpus with <unk>
    In the future replace this code with an averaging of the probability with the 
    five nearest neighbors that are in the corpus using fasttext word vectors'''
    s = set(text.split(" ")).difference(corpus_word_set)
    return re.sub(r"[\w'-]+", lambda m: '<unk>' if m.group() in s else m.group(), text)


# Set-up preprocess data-structures.
special_cases = ["pg.","e.g.","i.e.","var.","vs.","v.","sp.","a.m.","p.m.","pp.","al.","ssp.","him.","c.","o.","lit.",]
special_map = dict(zip(special_cases, ["HASHHASH"+str(abs(hash(z)))+"HASHHASH" for z in special_cases]))
# Efficiently map special cases to hash strings to avoid them from being altered.
rep = dict((re.escape(k), v) for k, v in special_map.items())
pattern = re.compile("|".join(rep.keys()))
# Reverse map data-structures
inv_map = {v: k for k, v in special_map.items()}
inv_rep = dict((re.escape(k), v) for k, v in inv_map.items())
inv_pattern = re.compile("|".join(inv_rep.keys()))


def preprocess(text):
    '''Prepares generic raw text for NLP models by handling punctuation and 
    seperating tokens in more meaningful ways.'''
    
    # Efficiently map special cases to hash strings to avoid them from being altered.
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)

    text = text + " " # Add trailing space to simplifiy pattern matching

    # Pad punctuation (note we could just add space around all punctuation then delete duplicate spaces, 
    # but this can remove newlines, so instead we pad left, then right, then both.)
    punctuation = r'([\[\]""~:;%#$^&*/\\\_!?{}()])'
    # LEFT Pad punctuation (except for ',', '-', and '.' with white space.
    text = re.sub(r'(\S)'+punctuation+r'([\s])', r'\1 \2\3', text, 0, re.IGNORECASE) 
    # RIGHT Pad punctuation (except for ',', '-', and '.' with white space.
    text = re.sub(r'(\s)'+punctuation+r'(\S)', r'\1\2 \3', text, 0, re.IGNORECASE) 
    # Pad punctuation between two words (except for ',', '-', and '.' with white space.
    text = re.sub(r'(\S)'+punctuation+r'(\S)', r'\1 \2 \3', text, 0, re.IGNORECASE) 
    # text = re.sub(r'\s{2,}', ' ', text) # Remove multiple adjacent spaces.

    # Prepend apostrophe's with white space on left side only e.g. what's --> what 's
    text = re.sub(r"(\S)(')(\S)", r'\1 \2\3', text, 0, re.IGNORECASE) 

    # replace the times 'x' between numbers with " x " padded with white space.
    text = re.sub(r"([0-9])x([0-9])", r"\1 x \2", text) 

    # Add left-space before non-numerical comma (we cheer, then yell --> we cheer , then yell)
    text = re.sub(r"([a-z0-9+])\,(\s)", r"\1 ,\2", text , 0, re.IGNORECASE) 

    # Handle those tricky periods. '.'
    text = re.sub(r"(\W[a-z]+)\.(\s)", r"\1 .\2", text)

    # Replace ',', '-', and '.' with '@,@', '@-@', and '@.@' where appropriate
    # DO THIS LAST
    text = re.sub(r"([\w])\-(\w)", r"\1 @-@ \2", text , 0, re.IGNORECASE)
    # replace decimals/commas in numbers with @.@ @,@ padded with white space.
    # ([0-9]) finds single digit, followed by (\.|\,) period or comma, followed by a single digit, 
    #followed by zero or more digits followed by non-alphanumeric. '(?=' finds all overlapping patterns.
    text = re.sub(r"([0-9])(\.|\,)([0-9])(?=[0-9]*\W)", r"\1 @\2@ \3", text , 0, re.IGNORECASE)
    # Map special cases back    
    text = inv_pattern.sub(lambda m: inv_rep[re.escape(m.group(0))], text)

    text = text.strip() # remove leading and trailing white space
    # text = re.sub(r"([a-z])\-([a-z])", r"\1 @-@ \2", text , 0, re.IGNORECASE)
    
    return text


def compute_nll(text, verbose = False):
    '''Computes the average negative log likelihood of the text. Assumes English. 
    Works best for a one to a few sentences. 
    Run seperately on groups of sentences instead of letting text be a large corpus.'''
    text = preprocess(text)
    text = replace_words_not_in_corpus_with_unk(text)
    corpus_idxs, ntokens = tokenize(text, corpus_dict)
    mydata_source = batchify(corpus_idxs, bsz = eval_batch_size ) #bsz is batch size
    probs = evaluate(mydata_source, ntokens)
    text_tokens = text.split(' ')
    running_sentence = text_tokens[0] # Add first word
    prob_first_word = word2probstart[running_sentence]
    nll = -np.log(prob_first_word if prob_first_word > 0 else 1e-50)
    for i in range(len(probs) - 1):
        next_word = text_tokens[i+1]
        idx = corpus_dict.word2idx[next_word]
        running_sentence += " " + next_word
        nll +=  -np.log(float(probs[i][0][idx]))
    avg_nll = nll / len(corpus_idxs)
    if verbose:
        print(running_sentence, " | avg NLL:", round(avg_nll, 2))
    return avg_nll



# consider adding cutoff thresholds for probabilities (test this out)

