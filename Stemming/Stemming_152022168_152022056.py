import csv
import re

class Stemmer:
    def __init__(self):
        self.kamus = self.load_kamus()
        self.stopwords = self.load_stopwords()

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
        # Remove -lah, -kah, -tah, -pun
        if word.endswith(('lah', 'kah', 'tah', 'pun')):
            word = re.sub(r'(lah|kah|tah|pun)$', '', word)
            
        # Remove -ku, -mu, -nya
        if word.endswith(('ku', 'mu', 'nya')):
            word = re.sub(r'(ku|mu|nya)$', '', word)
            
        return word

    def remove_derivation_suffixes(self, word):
        if word.endswith(('i', 'an', 'kan')):
            original = word
            
            # Remove -i, -an, or -kan
            if word.endswith('kan'):
                word = word[:-3]
            elif word.endswith(('i', 'an')):
                word = word[:-2]

            # Special case for -an where last letter is k
            if original.endswith('an') and word.endswith('k'):
                word = word[:-1]
                if self.check_kamus(word):
                    return word
                word = word + 'k'

            # Check if result exists in dictionary
            if self.check_kamus(word):
                return word
                
            # If not found, restore the original suffix
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
            return word

        if word[:2] in ['di', 'ke', 'se']:
            prefix = word[:2]
            stemmed = word[2:]
            
        elif word.startswith(('ter', 'bel')):
            prefix = word[:3]
            stemmed = word[3:]
            
        elif word.startswith(('me', 'pe', 'be')):
            if len(word) > 3:
                if word[2] == 'r' and word[3] in ['a','i','u','e','o']:
                    prefix = word[:3]
                    stemmed = word[3:]
                else:
                    prefix = word[:2]
                    stemmed = word[2:]
            else:
                prefix = word[:2]
                stemmed = word[2:]
        else:
            return word

        # Check if result exists in dictionary
        if self.check_kamus(stemmed):
            return stemmed
            
        # Try next iteration
        return self.remove_prefix(word, iteration + 1)

    def stem_word(self, word):
        if not self.is_valid_word(word):
            return word
            
        # Step 1: Check in dictionary
        if self.check_kamus(word):
            return word

        # Step 2: Remove inflection suffixes
        word = self.remove_inflection_suffixes(word)
        if self.check_kamus(word):
            return word

        # Step 3: Remove derivation suffixes
        word = self.remove_derivation_suffixes(word)
        if self.check_kamus(word):
            return word

        # Step 4: Remove prefixes
        word = self.remove_prefix(word)
        
        return word

    def stem_text(self, text):
        # Remove stopwords first
        text = self.remove_stopwords(text)
        
        # Split into words and stem each word
        words = text.split()
        stemmed_words = [self.stem_word(word) for word in words]
        
        return ' '.join(stemmed_words)

# Example usage
if __name__ == "__main__":
    stemmer = Stemmer()
    
    # Test cases
    test_texts = [
        "mendengarkan pembicaraan",
        "perjuangan kemerdekaan",
        "pembelajaran matematika",
        "perkuliahan mahasiswa"
    ]
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        print(f"Stemmed: {stemmer.stem_text(text)}")
