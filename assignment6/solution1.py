import sys
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk import pos_tag
from nltk import FreqDist
import string
import json
import math
import numpy.random
import itertools


class Solution1:
    def __init__(self,dictionary_file, training_file):
        self.dictionary = self.read_json_file(dictionary_file)
        training_data = self.read_text_file(training_file)
        
        # Prepare language models
        self.unigram_words = None
        self.bigram_words = None
        self.unigram_pos_words = None
        self.bigram_pos_words = None
        self.unigram_pos = None
        self.bigram_pos = None
        self.train(training_data)
        
    @staticmethod
    def read_text_file(filename):
        """
        Read the text file
        :param filename: filename of the text file
        :return: list of lines of the text file
        """
        try:
            file = open(filename, 'r')
        except:
            print('Cannot read file ' + filename + '. Please check the path', file=sys.stderr)
            sys.exit(1)
        output = []
        
        for line in file:
            line = line.strip().lower()
            output.append(line)
        return output
    
    @staticmethod
    def read_json_file(filename):
        """
        Read a json file
        :param filename: filename of the json file
        :return: dictionary object of json
        """
        try:
            file = open(filename, 'r')
        except:
            print('Cannot read file ' + filename + '. Please check the path', file=sys.stderr)
            sys.exit(1)
        return json.load(file)
        
    @staticmethod
    def words_to_sentence(words):
        return ''.join([word if word in string.punctuation else ' ' + word for word in words]).strip()
        
    @staticmethod
    def print_translation(title, source, translation):
        
        print('%s' % title)
        print('Original')
        print('%s' % source)
        print('Translated')
        print('%s' % translation)
        print('\n')

    def train(self, lines):
        """
        Training unigram, bigram, unigram with pos and bigram with pos models
        :param lines: Training lines
        """
        
        unigram_words = []
        bigram_words = []
        unigram_pos_words = []
        bigram_pos_words = []
        unigram_pos = []
        bigram_pos = []
        trigram_words = []
        
        for line in lines:
            words = word_tokenize(line)
            words_pos = pos_tag(words)
            pos = [word[1] for word in words_pos]
            unigram_words = unigram_words + ['<s>'] + words + ['</s>']
            unigram_pos_words = unigram_pos_words + words_pos
            unigram_pos = unigram_pos + ['<s>'] + pos + ['</s>']
            bigram_words = bigram_words + list(
                ngrams(words, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                       right_pad_symbol='</s>'))
            bigram_pos_words = bigram_pos_words + list(
                ngrams(words_pos, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                       right_pad_symbol='</s>'))
            bigram_pos = bigram_pos + list(
                ngrams(words_pos, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>',
                       right_pad_symbol='</s>'))
            
        self.unigram_words = FreqDist(unigram_words)
        self.bigram_words = FreqDist(bigram_words)
        self.unigram_pos_words = FreqDist(unigram_pos_words)
        self.bigram_pos_words = FreqDist(bigram_pos_words)
        self.unigram_pos = FreqDist(unigram_pos)
        self.bigram_pos = FreqDist(bigram_pos)
        self.trigram_words = FreqDist(trigram_words)
    
    def get_bigram_words_probability(self, words):
        probability = 0
        vocabulary_size = len(self.unigram_words)
        bigrams = list(ngrams(words, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        for bigram in bigrams:
            probability += math.log(self.bigram_words.freq(bigram) + 1) - math.log(
                self.unigram_words.freq(bigram[1]) + vocabulary_size)
        
        return probability

    def get_trigram_words_probability(self, words):
        probability = 0
        vocabulary_size = len(self.unigram_words)
        trigrams = list(ngrams(words, 3, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        for trigram in trigrams:
            probability += math.log(self.trigram_words.freq(trigram) + 1) - math.log(self.bigram_words.freq(trigram[1]) + vocabulary_size)
        
        return probability
    
    def get_bigram_pos_words_probability(self, words):
        words = pos_tag(words)
        probability = 0
        vocabulary_size = len(self.unigram_pos_words)
        bigrams = list(ngrams(words, 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        for bigram in bigrams:
            probability += math.log(self.bigram_pos_words.freq(bigram) + 1) - math.log(
                self.unigram_pos_words.freq(bigram[1]) + vocabulary_size)
            
        return probability

    
    def get_bigram_pos_probability(self, words):
        probability = 0
        vocabulary_size = len(self.unigram_pos)
        bigrams = list(ngrams(words, 2))
        for bigram in bigrams:
            probability += math.log(self.bigram_pos.freq(bigram) + 1) - math.log(
                self.unigram_pos.freq(bigram[1]) + vocabulary_size)
    
        return probability
    
    def get_highest_probability_permutation(self, words, method):
        max_probability = -math.inf
        selected = None
        permutation_count = math.factorial(len(words)) if len(words) < 5 else 100
        for _ in range(permutation_count):
            permutation = numpy.random.permutation(words)
            probability = getattr(self, method)(permutation)
            if probability > max_probability:
                max_probability = probability
                selected = permutation
                
        return selected
    
    def get_arrangement_with_pos_model(self, words):
        words_pos = [('', '<s>')] + pos_tag(words) + [('', '</s>')]
        length = len(words_pos)
        
        for index, word in enumerate(words_pos):
            words_window = words_pos[index : index + 4]
            
            max_probability = -math.inf
            selected = None
            permutations = itertools.permutations(words_window)
            for permutation in permutations:
                pos = [word[1] for word in permutation]
                probability = self.get_bigram_pos_probability(pos)
                if probability > max_probability:
                    max_probability = probability
                    selected = permutation
            
            words_pos[index] = selected[0]
            words_pos[index + 1] = selected[1]
            words_pos[index + 2] = selected[2]
            words_pos[index + 3] = selected[3]
            
            if index == length - 4:
                break;
        return [word[0] for word in words_pos]
    
    def swap_verb_after_noun(self, words):
        words_pos = pos_tag(words)
        length = len(words_pos)
        for index, word in enumerate(words_pos):
            if (word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or word[1] == 'NNPS') \
                and (words_pos[index + 1][1] == 'VB' or words_pos[index + 1][1] == 'VBD' \
                     or words_pos[index + 1][1] == 'VBG' or words_pos[index + 1][1] == 'VBN' \
                     or words_pos[index + 1][1] == 'VBP' or words_pos[index + 1][1] == 'VBZ'):
                temp_word = words_pos[index + 1];
                words_pos[index + 1] = words_pos[index]
                words_pos[index] = temp_word
        return [word[0] for word in words_pos]

    def swap_adjective_after_noun(self, words):
        words_pos = pos_tag(words)
        length = len(words_pos)
        for index, word in enumerate(words_pos):
            if (word[1] == 'NN' or word[1] == 'NNS' or word[1] == 'NNP' or word[1] == 'NNPS') \
                and (words_pos[index + 1][1] == 'JJ' or words_pos[index + 1][1] == 'JJS' or words_pos[index + 1][1] == 'JJR'):
                temp_word = words_pos[index + 1];
                words_pos[index + 1] = words_pos[index]
                words_pos[index] = temp_word
        return [word[0] for word in words_pos]

    def translate(self, line):
        words = word_tokenize(line)
        translated_words = []
        for i, word in enumerate(words):
            if word not in string.punctuation:
                translated_words.append(self.dictionary[word])
            else:
                translated_words.append(word)
        
        translated_sentence = self.words_to_sentence(translated_words)
        self.print_translation('Normal Translation', line, translated_sentence)

        # Improvement 1: Swap the nearest verb with the word after noun
        translated_words = self.swap_verb_after_noun(translated_words)
        translated_sentence = self.words_to_sentence(translated_words)
        self.print_translation('Swap the nearest verb with the word after noun', line, translated_sentence)

        # Improvement 2: Swap the nearest adjective with the word after noun
        translated_words = self.swap_adjective_after_noun(translated_words)
        translated_sentence = self.words_to_sentence(translated_words)
        self.print_translation('Swap the nearest adjective with the word after noun', line, translated_sentence)
        
        # Improvement 3: Bigram Language Model
        selected_translation = self.get_highest_probability_permutation(translated_words, 'get_bigram_words_probability')
        translated_sentence = self.words_to_sentence(selected_translation)
        self.print_translation('Translation with Bigram', line, translated_sentence)

        # Improvement 4: Trigram Language Model
        selected_translation = self.get_highest_probability_permutation(translated_words, 'get_trigram_words_probability')
        translated_sentence = self.words_to_sentence(selected_translation)
        self.print_translation('Translation with Trigram', line, translated_sentence)

        # Improvement 5: Bigram POS Language Model
        selected_translation = self.get_highest_probability_permutation(translated_words, 'get_bigram_pos_words_probability')
        translated_sentence = self.words_to_sentence(selected_translation)
        self.print_translation('Translation with Bigram and POS Tagging', line, translated_sentence)
        
        # Improvement 6: Rearrangement of POS
        selected_translation = self.get_arrangement_with_pos_model(translated_words)
        translated_sentence = self.words_to_sentence(selected_translation)
        self.print_translation('Translation with POS arrangment', line, translated_sentence)

    def execute(self, input_file):
        lines = self.read_text_file(input_file)
        for line in lines:
            self.translate(line)

solution1 = Solution1('data/dictionary.json', 'data/sentences_en_dev.txt')
solution1.execute('data/sentences_es_dev.txt')
solution1.execute('data/sentences_es_test.txt')

