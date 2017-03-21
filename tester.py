from RNNNumpy import RNNNumpy

# data I/O
data = open('input.txt', 'r').read()  # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# hyperparameters
hidden_size = 100  # size of hidden layer of neurons
seq_length = 25  # number of steps to unroll the RNN for; we set this to a small number so we do BPTT exactly
learning_rate = 1e-1

# prepare data
X = [[char_to_ix[c] for c in data[i:i + seq_length]] for i in range(len(data) - seq_length)]
Y = [[char_to_ix[c] for c in data[i + 1:i + seq_length + 1]] for i in range(len(data) - seq_length)]

rnn = RNNNumpy(vocab_size, hidden_size)
rnn.init_h()
rnn.init_params()

rnn.sgd(X, Y, eta=learning_rate, element_map=ix_to_char)
