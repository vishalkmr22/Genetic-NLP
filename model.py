import numpy as np
from collections import defaultdict
import pickle
import time as tm
import warnings
import os

# Helper function to extract bigrams
def get_bigrams(word, lim=None):
    # Get all bigrams
    bg = map(''.join, list(zip(word, word[1:])))
    # Remove duplicates and sort them
    bg = sorted(set(bg))
    # Make them into an immutable tuple and retain only the first few
    return tuple(bg)[:lim]

# Define the tree node structure
class TreeNode:
    def __init__(self):
        self.children = {}
        self.is_leaf = False
        self.words = []
        self.guesses = []
        self.bigram = None

# Function to calculate entropy
def calculate_entropy(splits, total_words):
    entropy = 0
    for split in splits:
        p = len(split) / total_words
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

# Function to build the decision tree with lookahead
def build_tree(words, depth=0, max_depth=10, min_words=5, lookahead_depth=2):
    if depth == max_depth or len(words) <= min_words:
        node = TreeNode()
        node.is_leaf = True
        node.words = words
        node.guesses = words[:min(5, len(words))]
        return node

    bigram_counts = defaultdict(int)
    for word in words:
        bigrams = get_bigrams(word)
        for bigram in bigrams:
            bigram_counts[bigram] += 1

    best_bigram = None
    best_entropy = float('inf')
    total_words = len(words)
    for bigram in bigram_counts:
        left_split = [word for word in words if bigram in get_bigrams(word)]
        right_split = [word for word in words if bigram not in get_bigrams(word)]
        entropy = calculate_entropy([left_split, right_split], total_words)
        if entropy < best_entropy:
            best_entropy = entropy
            best_bigram = bigram

    if not best_bigram:
        node = TreeNode()
        node.is_leaf = True
        node.words = words
        node.guesses = words[:min(5, len(words))]
        return node

    node = TreeNode()
    node.bigram = best_bigram
    left_split = [word for word in words if best_bigram in get_bigrams(word)]
    right_split = [word for word in words if best_bigram not in get_bigrams(word)]

    if lookahead_depth > 0:
        left_child = build_tree(left_split, depth + 1, max_depth, min_words, lookahead_depth - 1)
        right_child = build_tree(right_split, depth + 1, max_depth, min_words, lookahead_depth - 1)
    else:
        left_child = build_tree(left_split, depth + 1, max_depth, min_words, 0)
        right_child = build_tree(right_split, depth + 1, max_depth, min_words, 0)

    node.children['left'] = left_child
    node.children['right'] = right_child
    return node

# Function to fit the model
def my_fit(words):
    return build_tree(words)

# Function to predict based on the model
def my_predict(model, bigram_list):
    node = model
    while not node.is_leaf:
        if node.bigram in bigram_list:
            node = node.children['left']
        else:
            node = node.children['right']
    return node.guesses

# Main script
if __name__ == "__main__":
    with open("dict_secret", 'r') as f:
        words = f.read().split('\n')[:-1]  # Omit the last line since it is empty
        num_words = len(words)

    n_trials = 5
    t_train = 0
    m_size = 0
    t_test = 0
    prec = 0

    lim_bg = 5
    lim_out = 5

    for t in range(n_trials):
        tic = tm.perf_counter()
        model = my_fit(words)
        toc = tm.perf_counter()
        t_train += toc - tic

        with open(f"model_dump_{t}.pkl", "wb") as outfile:
            pickle.dump(model, outfile, protocol=pickle.HIGHEST_PROTOCOL)

        m_size += os.path.getsize(f"model_dump_{t}.pkl")

        tic = tm.perf_counter()

        for (i, word) in enumerate(words):
            bg = get_bigrams(word, lim=lim_bg)
            guess_list = my_predict(model, bg)

            # Do not send long guess lists -- they will result in lower marks
            guess_len = len(guess_list)
            # Ignore all but the first 5 guesses
            guess_list = guess_list[:lim_out]

            # Notice that if 10 guesses are made, one of which is correct,
            # score goes up by 1/10 even though only first 5 guesses are considered
            # Thus, it is never beneficial to send more than 5 guesses
            if word in guess_list:
                prec += 1 / guess_len

        toc = tm.perf_counter()
        t_test += toc - tic

    t_train /= n_trials
    m_size /= n_trials
    t_test /= n_trials
    prec /= (n_trials * num_words)

    print(t_train, m_size, prec, t_test)