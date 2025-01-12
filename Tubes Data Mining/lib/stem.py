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
        # Add rules for compound words
        self.compound_rules = {
            'ber': {'ajar': 'ajar'},  # contoh: belajar -> ajar
            'ke': {'tahu': 'tahu'},   # contoh: ketahui -> tahu
        }
        # Add rules for repeated words
        self.repeated_markers = ['-', '2']  # untuk kata ulang seperti "jalan-jalan" atau "jalan2"

    def load_kamus(self):
        with open('documents/Kamus.txt', 'r') as file:
            return set(word.strip().lower() for word in file)

    def load_stopwords(self):
        stopwords = set()
        with open('documents/Stopword.txt', 'r') as file:
            for line in file:
                stopwords.add(line.strip().lower())
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
            elif word.endswith('an'):
                word = word[:-2]
            elif word.endswith('i'):
                word = word[:-1]

            if self.check_kamus(word):
                return word

            if original.endswith('an') and word.endswith('k'):
                word = word[:-1]
                if self.check_kamus(word):
                    return word
                word = word + 'k'

            return word  # Return the modified word instead of the original
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
            if prefix == 'be' and stemmed.startswith('r'):
                stemmed = stemmed[1:]
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

        next_word, next_prefix = self.remove_prefix(stemmed, iteration + 1)
        return next_word, next_prefix or prefix_type

    def handle_repeated_word(self, word):
        """Handle repeated words like 'jalan-jalan' or 'jalan2'"""
        for marker in self.repeated_markers:
            if marker in word:
                parts = word.split(marker)
                if len(parts) == 2 and parts[0] == parts[1]:
                    return parts[0]
        return word

    def recode_prefix(self, prefix, word):
        """Enhanced prefix recoding rules based on Asian's modifications"""
        if prefix in ['me', 'pe']:
            if word.startswith('ng'):
                return word[2:]
            elif word.startswith('ny'):
                return 's' + word[2:]
            elif word.startswith('n'):
                if word[1] in ['d', 't']:  # Enhanced rule for 'me-' type
                    return word[1:]
                return 't' + word[1:]
            elif word.startswith('m'):
                if word[1] in ['b', 'p']:  # Enhanced rule for 'me-' type
                    return word[1:]
                return word[1:]  # Remove 'm' for 'me-' or 'pe-' type
            elif word.startswith('l') and len(word) > 1:
                return word[1:]
        return word

    def stem_word(self, word):
        steps = []
        
        # Handle repeated words first
        original = word
        word = self.handle_repeated_word(word)
        if word != original:
            steps.append(f"Found repeated word: '{original}' -> '{word}'")
        
        if not self.is_valid_word(word):
            steps.append(f"Invalid word: '{word}'")
            return word, steps

        # Step 1: Dictionary check
        steps.append(f"Step 1: Checking '{word}' in dictionary")
        if self.check_kamus(word):
            steps.append(f"Word '{word}' found in dictionary")
            return word, steps

        original_word = word

        # Check special cases for combined prefix-suffix
        current_prefix = self.get_prefix_type(word)
        current_suffix = None
        for suffix in ['i', 'an', 'kan', 'lah', 'kah', 'tah', 'pun', 'ku', 'mu', 'nya']:
            if word.endswith(suffix):
                current_suffix = suffix
                break

        if current_prefix and current_suffix:
            steps.append(f"Found prefix '{current_prefix}' and suffix '{current_suffix}'")

        # Normal processing order
        # Step 2: Remove inflectional suffixes
        steps.append(f"Step 2: Checking inflection suffixes in '{word}'")
        temp_word = self.remove_inflection_suffixes(word)
        if temp_word != word:
            steps.append(f"Removed inflection suffix: '{word}' -> '{temp_word}'")
            if self.check_kamus(temp_word):
                steps.append(f"Found word '{temp_word}' in dictionary after inflection removal")
                return temp_word, steps
            word = temp_word

        # Step 3: Remove derivational suffixes
        steps.append(f"Step 3: Checking derivation suffixes in '{word}'")
        temp_word = self.remove_derivation_suffixes(word)
        if temp_word != word:
            steps.append(f"Removed derivation suffix: '{word}' -> '{temp_word}'")
            if self.check_kamus(temp_word):
                steps.append(f"Found word '{temp_word}' in dictionary after derivation removal")
                return temp_word, steps
            word = temp_word

        # Step 4: Remove prefixes
        steps.append(f"Step 4: Checking prefixes in '{word}'")
        word, prefix_type = self.remove_prefix(word)
        if prefix_type:
            steps.append(f"Removed prefix {prefix_type}: -> '{word}'")

        # Final dictionary check
        if self.check_kamus(word):
            steps.append(f"Found final stemmed word '{word}' in dictionary")
            return word, steps

        steps.append(f"No root word found, returning original word '{original_word}'")
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
        # First tokenize the text
        tokens = self.tokenize(text)
        
        # Filter out stopwords before stemming
        tokens = [t for t in tokens if t['token'].lower() not in self.stopwords]
        
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
    
    print("=== Testing Stemmer ===")
    print(f"Original sentence: {test_sentence}\n")
    
    # Using stem_text but first get tokens
    tokens = stemmer.tokenize(test_sentence)
    
    # Filter stopwords but keep all other words
    filtered_tokens = [t for t in tokens if t['token'].lower() not in stemmer.stopwords]
    
    print("Processing each word (excluding stopwords):")
    print("-" * 50)
    for token in filtered_tokens:
        stemmed_word, steps = stemmer.stem_word(token['token'])
        print(f"\nOriginal: {token['token']}")
        print(f"Stemmed:  {stemmed_word}")
        print("Steps:")
        for step in steps:
            print(f"- {step}")
        print("-" * 50)
    
    # Still show the final stemmed text
    stemmed_text, _ = stemmer.stem_text(test_sentence)
    print("\nFinal stemmed text:")
    print(f"Result: {stemmed_text}")