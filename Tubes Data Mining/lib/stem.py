import re
import string

class Stemmer:
    def __init__(self):
        self.kamus = self.load_kamus()
        self.stopwords = self.load_stopwords()
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
        self.repeated_markers = ['-', '2']
        self.suffix_types = {
            'lah': '-lah',
            'kah': '-kah',
            'tah': '-tah',
            'pun': '-pun',
            'ku': '-ku',
            'mu': '-mu',
            'nya': '-nya',
            'i': '-i',
            'an': '-an',
            'kan': '-kan'
        }

    def load_kamus(self):
        with open('documents/Kamus.txt', 'r') as file:
            return set(word.strip().lower() for word in file)

    def load_stopwords(self):
        with open('documents/Stopword.txt', 'r') as file:
            return set(line.strip().lower() for line in file)

    def is_valid_word(self, word):
        return bool(re.match('^[a-zA-Z]+$', word)) and len(word) > 1

    def check_kamus(self, word):
        return word.lower() in self.kamus

    def remove_inflection_suffixes(self, word):
        original = word
        suffix_found = None

        if word.endswith(('lah', 'kah', 'tah', 'pun')):
            for suffix in ['lah', 'kah', 'tah', 'pun']:
                if word.endswith(suffix):
                    word = word[:-len(suffix)]
                    suffix_found = self.suffix_types[suffix]
                    break
        
        if word.endswith(('ku', 'mu', 'nya')):
            for suffix in ['ku', 'mu', 'nya']:
                if word.endswith(suffix):
                    word = word[:-len(suffix)]
                    suffix_found = self.suffix_types[suffix]
                    break

        return word, suffix_found

    def remove_derivation_suffixes(self, word):
        original = word
        suffix_found = None

        if word.endswith(('i', 'an', 'kan')):
            if word.endswith('kan'):
                word = word[:-3]
                suffix_found = self.suffix_types['kan']
            elif word.endswith('an'):
                word = word[:-2]
                suffix_found = self.suffix_types['an']
            elif word.endswith('i'):
                word = word[:-1]
                suffix_found = self.suffix_types['i']

            if self.check_kamus(word):
                return word, suffix_found

            if original.endswith('an') and word.endswith('k'):
                word = word[:-1]
                if self.check_kamus(word):
                    return word, suffix_found
                word = word + 'k'
            
            return word, suffix_found
        return word, None

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
        for marker in self.repeated_markers:
            if marker in word:
                parts = word.split(marker)
                if len(parts) == 2 and parts[0] == parts[1]:
                    return parts[0]
        return word

    def recode_prefix(self, prefix, word):
        if prefix in ['me', 'pe']:
            if word.startswith('ng'):
                return word[2:]
            elif word.startswith('ny'):
                return 's' + word[2:]
            elif word.startswith('n'):
                if word[1] in ['d', 't', 'c']:
                    return word[1:]
            elif word.startswith('m'):
                if word[1] in ['b', 'p']:
                    return word[1:]
            elif word.startswith('l') and len(word) > 1:
                return word[1:]
        return word

    def stem_word(self, word):
        steps = []
        word = self.handle_repeated_word(word)
        
        if not self.is_valid_word(word):
            return word, steps

        if self.check_kamus(word):
            return word, steps

        original_word = word

        # Remove inflection suffixes
        temp_word, inflection_suffix = self.remove_inflection_suffixes(word)
        if temp_word != word:
            if inflection_suffix:
                steps.append(f"Removed inflection suffix {inflection_suffix}")
            if self.check_kamus(temp_word):
                return temp_word, steps
            word = temp_word

        # Remove derivation suffixes
        temp_word, derivation_suffix = self.remove_derivation_suffixes(word)
        if temp_word != word:
            if derivation_suffix:
                steps.append(f"Removed derivation suffix {derivation_suffix}")
            if self.check_kamus(temp_word):
                return temp_word, steps
            word = temp_word

        # Remove prefix (existing code)
        word, prefix_type = self.remove_prefix(word)
        if prefix_type:
            steps.append(f"Removed prefix {prefix_type}")

        if self.check_kamus(word):
            return word, steps

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
        text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        for p in self.punctuation:
            text = text.replace(p, ' ')
        text = re.sub(r'[\u200b\u200c\u200d\ufeff\xa0]', ' ', text)
        text = ' '.join(text.split())
        words = text.split()
        tokens = []
        position = 0
        for word in words:
            if word and self.is_valid_word(word):
                tokens.append({
                    'token': word,
                    'position': position,
                    'original': word
                })
            position += 1
        return tokens

    def stem_text(self, text):
        tokens = self.tokenize(text)
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