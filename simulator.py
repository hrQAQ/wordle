import os
import numpy as np
import math
import json

MISS = np.uint8(0)  # Missed
MISPLACED = np.uint8(1) # Correct letter, wrong position
EXACT = np.uint8(2) # Correct letter, correct position

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
)
SHORT_WORD_LIST_FILE = os.path.join(DATA_DIR, "possible_words.txt")
LONG_WORD_LIST_FILE = os.path.join(DATA_DIR, "allowed_words.txt")
WORD_FREQ_FILE = os.path.join(DATA_DIR, "wordle_words_freqs_full.txt")
WORD_FREQ_MAP_FILE = os.path.join(DATA_DIR, "freq_map.json")
SECOND_GUESS_MAP_FILE = os.path.join(DATA_DIR, "second_guess_map.json")
PATTERN_MATRIX_FILE = os.path.join(DATA_DIR, "pattern_matrix.npy")
ENT_SCORE_PAIRS_FILE = os.path.join(DATA_DIR, "ent_score_pairs.json")

# To store the large grid of patterns at run time
PATTERN_GRID_DATA = dict()

# To avoid log(0)
def safe_log2(x):
    return math.log2(x) if x > 0 else 0

# Reading from files
def get_word_list(short=False):
    result = []
    file = SHORT_WORD_LIST_FILE if short else LONG_WORD_LIST_FILE
    with open(file) as fp:
        result.extend([word.strip() for word in fp.readlines()])
    return result

def get_word_frequencies(regenerate=False):
    if os.path.exists(WORD_FREQ_MAP_FILE) and not regenerate:
        with open(WORD_FREQ_MAP_FILE) as fp:
            result = json.load(fp)
        return result
    # Otherwise, regenerate
    freq_map = dict()
    with open(WORD_FREQ_FILE) as fp:
        for line in fp.readlines():
            pieces = line.split(' ')
            word = pieces[0]
            freqs = [
                float(piece.strip())
                for piece in pieces[1:]
            ]
            freq_map[word] = np.mean(freqs[-5:])    # Last 5 years
    with open(WORD_FREQ_MAP_FILE, 'w') as fp:
        json.dump(freq_map, fp)
    return freq_map