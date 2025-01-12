import csv
import re
from tkinter import filedialog
import tkinter as tk
import PyPDF2
from docx import Document
import os
import string
import math
from collections import defaultdict

class Stemmer:
    def __init__(self):
        self.kamus = self.load_kamus()
        self.stopwords = self.load_stopwords()
        self.forbidden_combinations = {
            'be': ['i'],
            'di': ['an'],
            'ke': ['i', 'kan'],
            'me': ['an'],
            'se': ['i', 'kan']
        }
        self.punctuation = (string.punctuation + '"' + '"' + ''' + ''' + '—' + '–' + 
                          '•' + '·' + '⋅' + '∙' + '‧' + '・' + '･' + '►' + '▪' + '○' + 
                          '●' + '♦' + '■' + '★' + '☆' + '✓' + '✔' + '❖')
        self.prefix_types = {
            'di': 'di-',
            'ke': 'ke-',
            'se': 'se-',
            'te': 'te-',
            'ter': 'ter-',
            'me': 'me-',
            'be': 'be-',
            'pe': 'pe-'
        }

    def load_kamus(self):
        with open('documents/Kamus.txt', 'r') as file:
            return set(word.strip().lower() for word in file)

    def load_stopwords(self):
        stopwords = set()
        with open('documents/Stopword.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                stopwords.add(row[0].lower())
        return stopwords

    def is_valid_word(self, word):
        return bool(re.match('^[a-zA-Z]+$', word))

    def remove_stopwords(self, text):
        words = text.lower().split()
        return ' '.join([word for word in words if word not in self.stopwords])

    def check_kamus(self, word):
        return word.lower() in self.kamus

    def remove_inflection_suffixes(self, word):
        
        if word.endswith(('lah', 'kah', 'tah', 'pun')):
            word = re.sub(r'(lah|kah|tah|pun)$', '', word)
            
        
        if word.endswith(('ku', 'mu', 'nya')):
            word = re.sub(r'(ku|mu|nya)$', '', word)
            
        return word

    def remove_derivation_suffixes(self, word):
        if word.endswith(('i', 'an', 'kan')):
            original = word
            if word.endswith('kan'):
                word = word[:-3]
            elif word.endswith(('i', 'an')):
                word = word[:-2]

            if original.endswith('an') and word.endswith('k'):
                word = word[:-1]
                if self.check_kamus(word):
                    return word
                word = word + 'k'

            if self.check_kamus(word):
                return word
            return original
        return word

    def is_forbidden_combination(self, prefix, suffix):
        forbidden = {
            'be': ['i'],
            'di': ['an'],
            'ke': ['i', 'kan'],
            'me': ['an'],
            'se': ['i', 'kan']
        }
        return prefix in forbidden and suffix in forbidden[prefix]

    def remove_prefix(self, word, iteration=1):
        if iteration > 3:
            return word, None  

        previous_prefix = None if iteration == 1 else self.get_prefix_type(word)

        if word[:2] in ['di', 'ke', 'se']:
            prefix = word[:2]
            stemmed = word[2:]
            prefix_type = self.prefix_types[prefix]
        elif word.startswith('ter'):
            prefix = 'ter'
            stemmed = word[3:]
            prefix_type = 'ter-'
        elif word.startswith('te'):
            prefix = 'te'
            stemmed = word[2:]
            prefix_type = 'te-'
        elif word.startswith(('me', 'pe', 'be')):
            if len(word) > 3 and word[2] == 'r' and word[3] in 'aiueo':
                prefix = word[:3]
                stemmed = word[3:]
            else:
                prefix = word[:2]
                stemmed = word[2:]
            prefix_type = self.prefix_types[prefix[:2]]
        else:
            return word, None

        if previous_prefix == prefix_type:
            return word, None

        if self.check_kamus(stemmed):
            return stemmed, prefix_type

        recoded = self.recode_prefix(prefix, stemmed)
        if recoded != stemmed and self.check_kamus(recoded):
            return recoded, prefix_type

        next_word, next_prefix = self.remove_prefix(word, iteration + 1)
        return next_word, next_prefix or prefix_type

    def recode_prefix(self, prefix, word):
        """Handle special recoding cases"""
        if prefix in ['me', 'pe']:
            if word.startswith('ng'):
                return word[2:]  
            elif word.startswith('ny'):
                return 's' + word[2:]  
            elif word.startswith('n'):
                return 't' + word[1:]  
        return word

    def stem_word(self, word):
        steps = []
        if not self.is_valid_word(word):
            return word, steps

        steps.append(f"Step 1: Checking '{word}' in dictionary")
        if self.check_kamus(word):
            steps.append("Result: Found in dictionary")
            return word, steps

        original_word = word
        suffix_removed = False
        prefix_type = None

        steps.append("Step 2: Checking inflection suffixes")
        if any(word.endswith(suffix) for suffix in ['lah', 'kah', 'tah', 'pun', 'ku', 'mu', 'nya']):
            old_word = word
            word = self.remove_inflection_suffixes(word)
            steps.append(f"Removed inflection suffix: {old_word} → {word}")

        steps.append(f"Step 3: Checking derivation suffixes for '{word}'")
        if any(word.endswith(suffix) for suffix in ['i', 'an', 'kan']):
            temp_word = word
            if word.endswith('kan'):
                word = word[:-3]
                suffix_removed = 'kan'
                steps.append(f"Removed -kan: '{temp_word}' → '{word}'")
            elif word.endswith('i'):
                word = word[:-1]
                suffix_removed = 'i'
                steps.append(f"Removed -i: '{temp_word}' → '{word}'")
            elif word.endswith('an'):
                word = word[:-2]
                suffix_removed = 'an'
                steps.append(f"Removed -an: '{temp_word}' → '{word}'")

                if word.endswith('k'):
                    k_word = word[:-1]
                    steps.append(f"Checking k-removal: '{word}' → '{k_word}'")
                    if self.check_kamus(k_word):
                        steps.append(f"Result: Found '{k_word}' in dictionary")
                        return k_word, steps
                    word = temp_word
                    steps.append("Restored original: k-removal unsuccessful")

            if self.check_kamus(word):
                steps.append(f"Result: Found '{word}' in dictionary")
                return word, steps
            word = temp_word
            steps.append("Restored original: suffix removal unsuccessful")

        if suffix_removed:
            steps.append("Step 4a: Checking prefix-suffix combinations")
            prefix = self.get_prefix_type(word)
            if prefix and suffix_removed in self.forbidden_combinations.get(prefix, []):
                steps.append(f"Found forbidden combination: {prefix}- with -{suffix_removed}")
                return original_word, steps

        steps.append("Step 4b: Removing prefixes")
        word, prefix_type = self.remove_prefix(word)
        if prefix_type:
            steps.append(f"Removed prefix type: {prefix_type}")

        if prefix_type:
            steps.append("Step 5: Checking recoding rules")
            recoded = self.recode_prefix(prefix_type.rstrip('-'), word)
            if recoded != word:
                steps.append(f"Applied recoding: {word} → {recoded}")
                word = recoded

        steps.append("Step 6: No root word found, returning original word")
        return original_word, steps

    def get_prefix_type(self, word):
        if word.startswith(('di', 'ke', 'se')):
            return word[:2]
        elif word.startswith(('ter', 'bel')):
            return word[:3]
        elif word.startswith(('me', 'pe', 'be')):
            return word[:2]
        return None

    def tokenize(self, text):
        """
        Tokenize text into words while handling punctuation and special cases.
        Returns list of tokens and their positions.
        """
        text = text.lower()
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\r', ' ')
        for p in self.punctuation:
            text = text.replace(p, ' ')
        text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0]', ' ', text)
        text = ' '.join(text.split())
        words = text.split()
        tokens = []
        position = 0
        for word in words:
            if (word and 
                self.is_valid_word(word)):
                tokens.append({
                    'token': word,
                    'position': position,
                    'original': word
                })
            position += 1
        return tokens

    def stem_text(self, text):
        tokens = self.tokenize(text)
        tokens = [t for t in tokens if t['token'] not in self.stopwords]
        results = []
        all_steps = []
        for token in tokens:
            stemmed_word, steps = self.stem_word(token['token'])
            token['stemmed'] = stemmed_word
            results.append(stemmed_word)
            if token['token'] != stemmed_word:  
                all_steps.append((token['original'], stemmed_word, steps))
        return ' '.join(results), all_steps

# Test section
if __name__ == "__main__":
    stemmer = Stemmer()
    
    # Test sentence that will be split into words
    test_sentence = "ya tidak stemming kesamaan dihitung diambil indexnya pembelajaran mendengarkan berlarian"
    test_words = test_sentence.split()
    
    print("=== Testing Stemmer ===")
    print(f"Original sentence: {test_sentence}\n")
    print("Word by word stemming:")
    print("-" * 50)
    
    for word in test_words:
        stemmed, steps = stemmer.stem_word(word)
        print(f"\nOriginal: {word}")
        print(f"Stemmed:  {stemmed}")
        print("Steps:")
        for step in steps:
            print(f"- {step}")
        print("-" * 50)