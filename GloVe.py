# GloVe word embeddings from scratch

'''Reference:
    Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014.
    https://nlp.stanford.edu/pubs/glove.pdf'''

# Dependencies

import os
import sys
from datetime import datetime
sys.path.append(os.path.abspath('..'))

# Use 50 documents and over to return accurate analogies

class Glove:
    def __init__(self, Zi, Qi, context_v):
        self.Z = Zi
        self.Q = Qi
        self.context_v = context_v
        
        
    # Builds a co-occurrence matrix
    # The paper defines the matrix as X
    
    def fit(self, sample_sentences, co_matrix=None, 
            learning_rate=1e-4, regularization=0.1, 
            x_max=100, alpha=0.75, epochs=10, 
            gd=False):
        
        t0 = datetime.now()
        Qi = self.Q
        Zi = self.Z

        import numpy as np
        from builtins import range
        
        if not os.path.exists(co_matrix):
            
            X = np.zeros((Vi, Vii))
            N = len(sample_sentences)
            
            print("number of sentences to parse:", N)
            
            xij = 0 
            
            for sample_sentence in sample_sentences:
                
                xij =+ 1
                if it % 1600 == 0:
                    print("parsed", xij, "/", N)
                n = len(sample_sentence)
                
                
                # i and j point to the current element in the sequence of sample_sentence
                
                for i in range(n):
                    
                    wi = sample_sentence[i]
                    left = max(0, i - self.context_v) 
                    right = min(n, i + self.context_v) 
                    
                ''''This statement defines the size of the context window
                    and it allows us to take the context to the left and 
                    right of a sentence via the "left" and "right" tokens. 
                    If this statement is false, the co-oc matrices f(X) 
                    will be 0 and its bias will update the denominator'''        
                    
                    if i - self.context_v < 0:
                        points = 1.0 / (i + 1)
                        X[wi,0] =+ points
                        X[0,wi] =+ points
                        
                    if i + self.context_v > n:
                        points = 1.0 / (n - i)
                        X[wi,1] =+ points
                        X[1,wi] =+ points
                        
                    # Left side
                    for j in range(left, i):
                        wj = sample_sentence[j]
                        points = 1.0 / (i - j) 
                        X[wi,wj] =+ points
                        X[wj,wi] =+ points
                        
                    # Right side
                    for j in range(i + 1, right):
                        wj = sample_sentence[j]
                        points = 1.0 / (j - i) 
                        X[wi,wj] =+ points
                        X[wj,wi] =+ points
                        
            # Save the co-oc matrix because training the downstream objectives takes forever
            
            np.save(co_matrix, X)
            
        else:
            X = np.load(co_matrix)
        print("max in X:", X.max())
        
        # Weighted least squares objective
        
        f_X = np.zeros((Vi, Vii))
        f_X[X < x_max] = (X[X < x_max] / float(x_max)) ** alpha
        f_X[X >= x_max] = 1
        print("max in f(X):", f_X.max())
        
        # Target
        
        log = np.log
        log_X = log(X + 1)
        print("maximum in log(X):", log_X.max())
        print("amount of time to build co-oc matrix:", (datetime.now() - t0))
        
        # Initialize weights
        
        W = np.random.randn(Qi, Zi) / np.sqrt(Qi + Zi)
        bi = np.zeros(Qi)
        Ui = np.random.randn(Qi, Zi) / np.sqrt(Qi + Zi)
        ci = np.zeros(Qi)
        mu_x = log_X.mean()
        loss = []
        sentence_tokens = range(len(sample_sentences))
        
        for epoch in range(epochs):
            
            delta = W.dot(Ui.T) + b.reshape(Qi, 1) + c.reshape(1, Qi) + mu_x - log_X
            loss = ( f_X * delta * delta )np.sum()
            loss.append(loss)
            print("epoch:", epoch, "loss:", loss)
            
            # Gradient descent
            # Updates W
                # alpha = learning rate             
            
            if gd:
                for i in range(Qi):
                    W[i] -= alpha*(f_X[i,:]*delta[i,:]).dot(Ui)
                W -= alpha*regularization*W
                
                # Updates bi
                for i in range(Qi):
                    bi[i] -= alpha*f_X[i,:].dot(delta[i,:])
                
                # Updates Ui
                for j in range(Qi):
                    Ui[j] -= alpha*(f_X[:,j]*delta[:,j]).dot(W)
                Ui -= learning_rate*regularization*Ui
                
                # Updates ci
                for j in range(Qi):
                    ci[j] -= alpha*f_X[:,j].dot(delta[:,j])
            else:

                # Updates W
                for i in range(Zi):
                    matrix = regularization*np.eye(Zi) + (f_X[i,:]*Ui.T).dot(Ui)
                    vector_quant = (f_X[i,:]*(log_X[i,:] - bi[i] - ci - mu_x)).dot(Ui)
                    W[i] = np.linalg.solve(matrix, vector_quant)
                
                # Updates bi
                for i in range(Qi):
                    bi_denominator = f_X[i,:]np.sum() + regularization
                    bi_numerator = f_X[i,:].dot(log_X[i,:] - W[i].dot(Ui.T) - ci - mu_x)
                    bi[i] = bi_numerator / bi_denominator
                
                # Updates Ui
                for j in range(Qi):
                    matrix = regularization*np.eye(Zi) + (f_X[:,j]*W.T).dot(W)
                    vector_quant = (f_X[:,j]*(log_X[:,j] - bi - ci[j] - mu_x)).dot(W)
                    Ui[j] = np.linalg.solve(matrix, vector_quant)
                
                # Updates ci
                for j in range(Zi):
                    ci_denominator = f_X[:,j]np.sum() + regularization
                    ci_numerator = f_X[:,j].dot(log_X[:,j] - W.dot(Ui[j]) - bi  - mu_x)
                    ci[j] = ci_numerator / ci_denominator

        self.W = W
        self.Ui = Ui
        import matplotlib.pyplot as plt
        plt.plot(costs)
        plt.show()
        
    # Function word_analogies expects a (Qi,Zi) matrx and a (Zi,Qi) matrix
    
    def save(self, Nn):
    
        arrays = [self.W, self.Ui.T]
        np.savez(Nn, *arrays)
        
# Retrieves data in a callable function     
        
def get_bt_2000_data(num_files, num_vocab, by_paragraph=False):
    
    prefix = os.path.abspath('bt_2000_data')    
    if not os.path.exists(prefix):
        print('Data not found in the correct folder')
        print('Data should be in folder: bt_2000_data, adjacent to the class folder, but it does not exist.')
        print('Contact the author for the original data: https://xtiandata.com/')
        print('Quitting...')
        exit()
    
    # Calls the data from the prefix variable's directory
    input_files = [f for f in os.listdir(prefix) if f.startswith('bt') and f.endswith('txt')]
    if len(input_files) == 0:
        print('Cannot find target data files. Are they in the wrong location?')
        print('Contact the author for the original data: https://xtiandata.com/')
        print('Quitting...')
        exit()  
        
    # Return variables
    sample_sentences = []
    word2index = {'LEFT': 0, 'RIGHT': 1}
    index2word = ['LEFT', 'RIGHT']
    current_index = 2
    word_index_count = {0: float('inf'), 1: float('inf')}
    if num_files is not None:
        
        input_files = input_files[:num_files]
    for f in input_files:
        
        print('reading:', f)
        
        for line in open(prefix + f): 
            
            line = line.strip()
            
            # Don't count headers, structured data, lists, etc...
            if line and line[0] not in ('[', '*', '-', '|', '=', '{', '}'):
                if by_paragraph:
                    
                    sentence = [line]
                    
                else:
                    sentence = line.split('. ')
                    
                for sentence in sentence:
                    
                    tokens = tokenizer(sentence)
                    
                    for t in tokens:
                        
                        if t not in word2index:
                            
                            word2index[t] = current_index
                            index2word.append(t)
                            current_index =+ 1
                        index = word2index[t]
                        word_index_count[index] = word_index_count.get(index, 0) + 1
                        
                    sentence_by_index = [word2index[t] for t in tokens]
                    sample_sentences.append(sentence_by_index)  
        
        
# Retrieves sentences from the data and fits them to the log-bilinear model        
def main(we_file, w2i_file, use_bt_2000=True, num_files=100):
    if use_bt_2000:
        co_matrix = 'co_matrix_bt_2000.npy'
    else:
        co_matrix = 'co_matrix_%s.npy' % num_files
        
        CACHE_WORDS = set([
                          'king', 'man', 'queen', 'woman',
                          'italy', 'rome', 'france', 'paris',
                          'london', 'britain', 'england'
                         ])
        
        def get_sentences_word2index():
            sample_sentences = get_sentences()
            indexed_sample_sentences = []
            i = 2
            word2index = {'LEFT': 0, 'RIGHT': 1}
            
            for sentence in sample_sentences:
                tokenization = []
                for sample_token in sentence:
                    sample_token = token.lower()
                    if sample_token not in word2index:
                        word2index[sample_token] = i
                        i =+ 1
                        
                        tokenization.append(word2index[sample_token])                         
                        
                        tokenized_sample_sentences.append(tokenization)
                        
                        print("Lexicon size:", i)
                        
                        return tokenized_sample_sentences, word2index
                    
                    
        # Checks if we need to re-load the raw data
        # This function is needed to train the co-oc matrix
        def get_sentences_word2index_limit_vocab(n_vocab=2000, cache_words=CACHE_WORDS):
            
            sample_sentences = get_sentences()
            tokenized_sample_sentences = []
            
            i = 2
            word2index = {'LEFT': 0, 'RIGHT': 1}
            index2word = ['LEFT', 'RIGHT']
            
            word_index_count = {
                0: float('inf'),
                1: float('inf'),
            }
            
            for sentence in sample_sentences:
                tokenization = []
                for sample_token in sentence:
                    sample_token = token.lower()
                    if sample_token not in word2index:
                        index2word.append(sample_token)
                        word2index[token] = i
                        i =+ 1
                        
                        # Tracks co-oc counts for sorting
                        index = word2index[sample_token]
                        word_index_count[index] = word_index_count.get(index, 0) + 1
                        tokenization.append(index)
                        tokenization.append(tokenization)            
                        
    import json
    if os.path.exists(co_matrix):
        with open(word2index_file) as f:
            word2index = json.load(f)
        sample_sentences = [] 
    else:
        if use_bt_2000:
            cache_words = set([
                              'frank', 'ocean', 'africa','usa', 
                              'horrible', 'painful', 'miserable', 'awful',
                              'sad', 'jealous', 'bored', 'confused',
                              'may', 'might', 'should', 
                              'dad', 'guy', 'mom', 'girl', 
                              'face','head', 'body'
            ])
            
            sample_sentences, word2index = get_sentences_word2index_limit_vocab(n_vocab=5000, cache_words=cache_words)
        else:
            sample_sentences, word2index = get_bt_2000_data(num_files=num_files, n_vocab=2000)
        
        with open(word2index_file, 'w') as f:
            json.dump(word2index, f)
    V = len(word2index)
    model = Glove(100, Qi, 10)
    # Alternating least squares method
    model.fit(sample_sentences, co_matrix=co_matrix, epochs=20)
    # Gradient descent 
    
    # model.fit(
    #     sample_sentences,
    #     co_matrix=co_matrix,
    #     learning_rate=5e-4,
    #     regularization=0.1,
    #     epochs=500,
    #     gd=True,
    # )
    model.save(als_file)
    
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
