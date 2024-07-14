# GloVe word embeddings from scratch
# Needs refactoring 

'''Reference:
    Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
    https://nlp.stanford.edu/pubs/glove.pdf'''

# Stdlib dependencies
import os
import json
from datetime import datetime
from builtins import range

import sys
sys.path.append(os.path.abspath('..'))

# Third party dependencies
import numpy as np
import matplotlib.pyplot as plt


class Glove:
    """
    This class implements the GloVe algorithm for learning word embeddings from a corpus of text. 
    The model builds a co-occurrence matrix that captures how frequently words co-occur with each 
    other in a context window. It then uses this matrix to train word and context vectors such that 
    their dot product approximates the logarithm of the word co-occurrence probabilities to produce
    word analogies.

    Attributes:
        embedding_dimension (int): Dimensionality of the embedding vectors.
        vocabulary_size (int): Size of the vocabulary.
        context_window_size (int): Size of the context window around each word.

    Methods:
        fit():
            Builds the co-occurrence matrix from the sample sentences and trains the GloVe model 
            using either gradient descent or alternating least squares.
        
        save(filename):
            Saves the trained word and context vectors to a file in .npz format.
    """
    def __init__(self, embedding_dimension, vocabulary_size, context_window_size):
        # Dimensionality of embedding vectors, Z matrix from the paper
        self.embedding_dimension = embedding_dimension
        
        # Size of the vocabulary, matrix Q from the paper
        self.vocabulary_size = vocabulary_size
        
        # Size of the context window
        self.context_window_size = context_window_size

    
    # Builds a co-occurrence matrix, X matrix from the paper
    def fit(self, sample_sentences, co_occurrence_matrix_file=None, 
            learning_rate=1e-4, regularization=0.1, 
            x_max=100, alpha=0.75, epochs=10, 
            gradient_descent=False):

        # Record start time for performance measurement
        start_time = datetime.now()
        # Embedding dimensionality
        embedding_dim = self.embedding_dimension
        # Vocabulary size
        vocab_size = self.vocabulary_size

        # Check if the co-occurrence matrix file exists
        if not os.path.exists(co_occurrence_matrix_file):
            # Initialize the co-occurrence matrix
            co_occurence_matrix = np.zeros((vocab_size, vocab_size))
            # Number of sentences
            num_sentences = len(sample_sentences)
            
            print("number of sentences to parse:", num_sentences)

            # Iterate over sentences
            for sentence_idx, sample_sentence in enumerate(sample_sentences):
                # Print progress every 1600 sentences
                if sentence_idx % 1600 == 0:
                    print("parsed", sentence_idx, "/", num_sentences)
                sentence_length = len(sample_sentence)                
                
                # Iterate over the words in the sentence
                for i in range(sentence_length):
                    # Current word
                    word_i = sample_sentence[i]
                    # Left context window
                    left_context = max(0, i - self.context_window_size)
                    # Right context window
                    right_context = min(sentence_length, i + self.context_window_size) 
                    
                ''''Defines the size of the context window
                    and it allows us to take the context to the left and 
                    right of a sentence via the "left" and "right" tokens. 
                    If this statement is false, the co-oc matrices f(X) 
                    will be 0 and its bias will update the denominator''' 
                    # Handle left boundary
                    if i - self.context_window_size < 0:
                        # Points for the boundary condition
                        points = 1.0 / (i + 1)
                        # Update co-occurance matrix
                        co_occurrence_matrix[word_i, 0] += points
                        # Update co-occurance matrix
                        co_occurrence_matrix[0, word_i] += points

                    # Handle right boundary
                    if i + self.context_window_size > sentence_length:
                        points = 1.0 / (sentence_length - i)
                        co_occurrence_matrix[word_i,1] += points
                        co_occurrence_matrix[1,word_i] += points
                        
                    # Left side
                    # Iterate over the left context
                    for j in range(left_context, i):
                        # Current context word
                        word_j = sample_sentence[j]
                        # Points for the co-occurrence
                        points = 1.0 / (i - j) 
                        co_occurrence_matrix[word_i, word_j] += points
                        co_occurrence_matrix[word_j, word_i] += points
                        
                    # Right side
                    for j in range(i + 1, right_context):
                        word_j = sample_sentence[j]
                        points = 1.0 / (j - i) 
                        co_occurrence_matrix[word_i, word_j] += points
                        co_occurrence_matrix[word_j, word_i] += points
                        
            # Save the co-oc matrix
            np.save(co_occurrence_matrix, X)
            
        else:
            # Load the co-occurrence matrix
            co_occurrence_matrix = np.load(co_occurrence_matrix)
        # Print the minimum value in the co-occurrence matrix
        print("max in co_occurrence_matrix:", co_occurrence_matrix.max())
        
        # Weighted least squares objective
        # Initialize the weighting function
        weighting_function = np.zeros(co_occurrence_matrix.shape)
        # Apply the weighting function
        weighting_function[co_occurrence_matrix < x_max] = (co_occurrence_matrix[co_occurrence_matrix < x_max] / float(x_max)) ** alpha
        weighting_function[co_occurrence_matrix >= x_max] = 1
        # Pring maximum value in the weighting function
        print("Max in weighting_function:", weighting_function.max())
        
        # Target
        # Compute the log of the co-occurrence matrix 
        log_co_occurrence_matrix = np.log(co_occurrence_matrix + 1)
        # Print the maximum value in the log co-occurrence matrix 
        print("Maximum in log(co_occurrence_matrix):", log_co_occurrence_matrix.max())
        # Print the time taken to build the co-occurrence matrix
        print("Amount of time to build co-occurence matrix:", (datetime.now() - start_time))
        
        # Initialize weights
        # Initialize word vectors
        W_word_vectors = np.random.randn(vocab_size, embedding_dim) / np.sqrt(vocab_size + embedding_dim)
        # Initialize word biases
        b_word_biases = np.zeros(vocab_size)
        # Initialize context vectors
        U_context_vectors = np.random.randn(vocab_size, embedding_dim) / np.sqrt(vocab_size + embedding_dim)
        # Initialize context biases
        c_context_biases = np.zeros(vocab_size)
        # Mean of the log co-occurrence matrix
        mu_log_co_occurrence = log_co_occurrence_matrix.mean()
        # Store the training loss
        training_loss = []

        # Iterate over epochs
        for epoch in range(epochs):
            delta = W_word_vectors.dot(U_context_vectors.T) + b_word_biases.reshape(vocab_size, 1) + c_context_biases.reshape(1, vocab_size) + mu_log_co_occurrence - log_co_occurrence_matrix
            # Compute loss for the epoch
            epoch_loss = (weighting_function * delta * delta ).sum()
            # Append each loss to the list
            training_loss.append(epoch_loss)
            print("epoch:", epoch, "loss:", epoch_loss)
            
            # Gradient descent             
            if gradient_descent:
                # Iterate over vocabulary size
                for i in range(vocab_size):
                    # Update word vectors using gradient descent
                    W_word_vectors[i] -= learning_rate * (weighting_function[i, :] * delta[i,:]).dot(U_context_vectors)
                # Regularization
                W_word_vectors -= learning_rate * regularization * W_word_vectors
                
                for i in range(V):
                    # Update word biases using gradient descent
                    b_word_biases[i] -= learning_rate * weighting_function[i, :].dot(delta[i,:])
                
                # Update context vectors using gradient descent & regularization
                for j in range(V):
                    U_context_vectors[j] -= learning_rate * (weighting_function[:, j] * delta[:, j]).dot(W)
                U_context_vectors -= learning_rate * regularization * U_context_vectors
                
                # # Update context biases using gradient descent
                for j in range(V):
                    c_context_biases[j] -= learning_rate * weighting_function[:, j].dot(delta[:, j])
            else:
                # Alternating least squares
                for i in range(V):
                    # Compute matrix for ALS
                    matrix = regularization * np.eye(embedding_dim) + (weighting_function[i, :] * U_context_vectors.T).dot(U_context_vectors)
                    # Compute vector for ALS
                    vector_quant = (weighting_function[i, :] * (log_co_occurrence_matrix[i, :] - b_word_biases[i] - c_context_biases  - mu_log_co_occurrence)).dot(U_context_vectors)
                    # Solve for word vectors
                    W_word_vectors[i] = np.linalg.solve(matrix, vector_quant)
                
                # Updates word biases
                for i in range(vocab_size):
                    # Compute denominator for word biases
                    bi_denominator = weighting_function[i, :].sum() + regularization
                    # Compute numerator for word biases
                    bi_numerator = weighting_function[i, :].dot(log_co_occurrence_matrix[i, :] - W_word_vectors[i].dot(U_context_vectors.T) - c_context_biases - mu_log_co_occurrence)
                    # Update word biases
                    b_word_biases[i] = bi_numerator / bi_denominator

                # Updates context vectors
                for j in range(vocab_size):
                    # Compute matrix for ALS
                    matrix = regularization * np.eye(embedding_dim) + (weighting_function[:, j] * W_word_vectors.T).dot(W_word_vectors)
                    # Compute vector for ALS
                    vector_quant = (weighting_function[:, j] * (log_co_occurrence_matrix[:, j] - b_word_biases  - c_context_biases[j] - mu_log_co_occurrence)).dot(W_word_vectors)
                    # Solve for context vectors
                    U_context_vectors[j] = np.linalg.solve(matrix, vector_quant)
                
                # Updates context biases
                for j in range(V):
                    # Compute denominator for context biases
                    ci_denominator = weighting_function[:, j].sum() + regularization
                    # Compute numerator for context biases
                    ci_numerator = weighting_function[:, j].dot(log_co_occurrence_matrix[:, j] - W_word_vectors.dot(U_context_vectors[j]) - b_word_biases   - mu_log_co_occurrence)
                    # Update context biases
                    c_context_biases[j] = ci_numerator / ci_denominator

        # Store the learned word vectors
        self.W_word_vectors = W_word_vectors
        # Store the learned context vectors
        self.U_context_vectors  = U_context_vectors 
        # Plot training loss
        plt.plot(training_loss)
        plt.show()

    
    # Save the learned embeddings
    def save(self, filename):
        arrays = [self.W_word_vectors, self.U_context_vectors.T]
        np.savez(filename, *arrays)


# Retrieves data from bt-2000 dataset     
def get_bt_2000_data(num_files, num_vocab, by_paragraph=False):
    # Path to data folder
    prefix = os.path.abspath('bt_2000_data')    
    # Check if folder exists
    if not os.path.exists(prefix):
        print('Data not found in the correct folder')
        print('Data should be in folder: bt_2000_data, adjacent to the class folder, but it does not exist.')
        print('Contact the author for the original data: https://xtiandata.com/')
        print('Quitting...')
        exit()
    
    # Calls the data from the prefix variable's directory
    input_files = [f for f in os.listdir(prefix) if f.startswith('bt') and f.endswith('txt')]
    # Check if there are no data files
    if len(input_files) == 0:
        print('Cannot find target data files. Are they in the wrong location?')
        print('Contact the author for the original data: https://xtiandata.com/')
        print('Quitting...')
        exit()  

    
    # List to store sample sentences
    sample_sentences = []
    # Dictionary to map words to indices
    word2index = {'LEFT': 0, 'RIGHT': 1}
    # List to map indices to words
    index2word = ['LEFT', 'RIGHT']
    # Current index for new words
    current_index = 2
    # Dictionary to count word occurrences
    word_index_count = {0: float('inf'), 1: float('inf')}
    
    if num_files is not None:
        # Limit the number of files to read
        input_files = input_files[:num_files]
        
    for f in input_files:
        # Print the current file being read
        print('reading:', f)
        for line in open(os.path.join(prefix + f)): 
            line = line.strip()
            # Don't count headers, structured data, lists, etc...
            if line and line[0] not in ('[', '*', '-', '|', '=', '{', '}'):
                # Check if paragraph is True
                if by_paragraph:
                    # Treat the whole line as a sentence
                    sentences = [line]
                else:
                    sentences = line.split('. ')
                for sentence in sentences:
                    tokens = sentence.split()
                    for t in tokens:
                        # Check if the token is not in the dictionary
                        if t not in word2index:
                            # Add the token to the dictionary
                            word2index[t] = current_index
                            # Add token to the list
                            index2word.append(t)
                            # Increment the current index
                            current_index =+ 1
                        # Get the index of the token
                        index = word2index[t]
                        # Increment the word count
                        word_index_count[index] = word_index_count.get(index, 0) + 1
                    # Convert tokens to indices
                    sentence_by_index = [word2index[t] for t in tokens]
                    # Add sentence to the list
                    sample_sentences.append(sentence_by_index)
    # Return sample sentences and word2index
    return sample_sentences, word2index


# Main function. Retrieves sentences from the data and fits them to the model        
def main(we_file, w2i_file, use_bt_2000=True, num_files=100):
    if use_bt_2000:
        # Co-occurrence matrix file for bt-2000 dataset
        co_occurrence_matrix_file = 'co_matrix_bt_2000.npy'
    else:
        # Co-occurrence matrix file for other datasets
        co_occurrence_matrix_file = 'co_matrix_%s.npy' % num_files
        
        # Cache words for limited vocabulary
        CACHE_WORDS = {
            'king', 'man', 'queen', 'woman',
            'italy', 'rome', 'france', 'paris',
            'london', 'britain', 'england'
        }

        
        def get_sentences_word2index():
            # Get sample sentences and word2index
            sample_sentences, word2index = get_bt_2000_data(num_files, num_vocab=2000)
            return tokenized_sample_sentences, word2index
                    
                    
        '''Checks if we need to re-load the raw data
            needed to train the co-oc matrix. Function 
            gets sentences with limited vocabulary'''
        def get_sentences_word2index_limit_vocab(n_vocab=2000, cache_words=CACHE_WORDS):
            sample_sentences, word2index = get_bt_2000_data(num_files, n_vocab=2000)
            # List to store tokenized sentences
            tokenized_sample_sentences = []
            # Current index for new words
            current_index = len(word2index)
            # Iterate over sample sentences
            for sentence in sample_sentences:
                # List to store tokenized sentence
                tokenization = []
                for token in sentence:
                    token = token.lower()
                    # Check if the token is not in the dictionary
                    if token not in word2index:
                        # Add token to the dictionary
                        word2index[token] = current_index
                        # Increment the current index
                        current_index += 1 
                    # Get the index of the token
                    index = word2index[token]
                    # Add the index to the tokenization
                    tokenization.append(index)
                # Add teh tokenized sentence to the list
                tokenized_sample_sentences.append(tokenization)
            return tokenized_sample_sentences, word2index
            

    # Check if the co-occurrence matrix file exists
    if os.path.exists(co_occurrence_matrix_file):
        with open(word2index_file) as f:
            word2index = json.load(f)
        sample_sentences = [] 
    else:
        # Check if bt-2000 dataset is to be used
        if use_bt_2000:
            # Cache words for limited vocabulary
            cache_words = {
                'frank', 'ocean', 'africa','usa', 
                'horrible', 'painful', 'miserable', 'awful',
                'sad', 'jealous', 'bored', 'confused',
                'may', 'might', 'should', 
                'dad', 'guy', 'mom', 'girl', 
                'face','head', 'body'
            }
            # Get sample sentences and word2index
            sample_sentences, word2index = get_sentences_word2index_limit_vocab(n_vocab=5000, cache_words=cache_words)
        else:
            sample_sentences, word2index = get_bt_2000_data(num_files=num_files, n_vocab=2000)

        # Open word2index file for writing
        with open(word2index_file, 'w') as f:
            json.dump(word2index, f)

    # Vocabulary size
    V = len(word2index)
    # Create the GloVe model
    model = Glove(100, V, 10)
    # Fit the model
    model.fit(sample_sentences, 
              co_occurrence_matrix_file=co_occurrence_matrix_file, 
              learning_rate=5e-4, 
              regularization=0.1, 
              epochs=20, 
              gradient_descent=True)
    # Save the model
    model.save(embedding_file)


# LEFT OFF HERE
# LEFT OFF HERE
# LEFT OFF HERE
# LEFT OFF HERE
# Computes analogies between vectors
def find_analogies(w1, w2, w3, We, word2index, index2word):
    Zi, Qi = We.shape
    dad = We[word2index[w1]]
    guy = We[word2index[w2]]
    mom = We[word2index[w3]]
    v0 = dad - guy + mom
    for dist in ('euclidean', 'cosine'):
        distances = pairwise_distances(v0.reshape(1, Qi), We, metric=dist).reshape(Zi)
        index = [x for x,y in sorted(enumerate(distances), key = lambda x: x[4])]
        best_index = -1
        keep_out = [word2index[w] for w in (w1, w2, w3)]
        for i in index:
            if i not in keep_out:
                best_index = i
                break
        analogous_word = index2word[best_index]
        print('closest match by', dist, 'distance:', analogous_word)
        print(w1, "-", w2, "=", analogous_word, "-", w3)
    
# Loads embedding analogies 
if __name__ == '__main__':
    we = 'glove_model.npz'
    w2i = 'glove_word2index.json'
    main(we, w2i, use_bt_2000=False)
    
    glove_model = np.load(we)
    word_one = glove_model['arr_0']
    word_two = glove_model['arr_1']
    with open(w2i) as f:
        word2index = json.load(f)
        index2word = {i:w for w,i in word2index.items()}
        
    for append in (True, False):
        print('** concat:', append)
        if append:
            We = np.hstack([word_one, word_two.T])
        else:
            We = (word_one + word_two.T) / 2
            
        find_analogies('king', 'man', 'woman', We, word2index, index2word)
        find_analogies('frank', 'ocean', 'africa', We, word2index, index2word)
        find_analogies('horrible', 'painful', 'miserable', We, word2index, index2word)
        find_analogies('sad', 'jealous', 'bored', We, word2index, index2word)
        find_analogies('may', 'might', 'should', We, word2index, index2word)
        find_analogies('dad', 'guy', 'mom', We, word2index, index2word)
        find_analogies('face', 'head', 'body', We, word2index, index2word)
