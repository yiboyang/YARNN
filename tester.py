from RNNNumpy import RNNNumpy


def test_char_lm():
    """Train a character level RNN language model"""
    # data I/O
    data = open('input.txt', encoding='utf8').read()  # should be simple plain text file
    chars = list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('data has %d characters, %d unique.' % (data_size, vocab_size))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}

    # hyperparameters
    hidden_size = 128  # size of hidden layer of neurons
    seq_length = 20  # number of steps to unroll the RNN for; we set this to a small number so we do BPTT exactly
    learning_rate = 1e-2

    # prepare data
    X = [[char_to_ix[c] for c in data[i:i + seq_length]] for i in range(len(data) - seq_length)]
    Y = [[char_to_ix[c] for c in data[i + 1:i + seq_length + 1]] for i in range(len(data) - seq_length)]

    rnn = RNNNumpy(vocab_size, hidden_size)
    rnn.init_h()
    rnn.init_params()

    rnn.sgd(X, Y, eta=learning_rate, adagrad=True, report_interval=1, element_map=ix_to_char)


def test_word_lm():
    """Train a word level RNN language model"""
    import nltk
    import itertools
    from util import pad_list
    # data I/O
    data = open('input.txt', encoding='utf8').read()  # should be simple plain text file
    sentences = nltk.sent_tokenize(data)  # get a list of sentences
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]  # tokenize the sentences into lists of words
    tokenized_sentences = [pad_list(wl, ' ') for wl in tokenized_sentences if
                           len(wl) > 1]  # insert spaces b/w words, throw away sentences with <= 1 word
    words = list(set(itertools.chain.from_iterable(tokenized_sentences)))

    data_size, vocab_size = len(data), len(words)
    print('data has %d words, %d unique.' % (data_size, vocab_size))
    w_to_ix = {w: i for i, w in enumerate(words)}
    ix_to_w = {i: w for i, w in enumerate(words)}

    # hyperparameters
    hidden_size = 128  # size of hidden layer of neurons
    seq_length = 20  # number of steps to unroll the RNN for; we set this to a small number so we do BPTT exactly
    learning_rate = 5e-2

    # prepare data
    X = [[w_to_ix[w] for w in sent[:-1]] for sent in tokenized_sentences]
    Y = [[w_to_ix[w] for w in sent[1:]] for sent in tokenized_sentences]

    rnn = RNNNumpy(vocab_size, hidden_size)
    rnn.init_h()
    rnn.init_params()

    rnn.sgd(X, Y, eta=learning_rate, adagrad=True, num_epochs=200, report_interval=1, element_map=ix_to_w)


# test_char_lm()
test_word_lm()
