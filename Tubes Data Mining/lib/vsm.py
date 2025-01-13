from collections import defaultdict
import math

class VSM:
    def __init__(self, stemmer):
        self.stemmer = stemmer
        self.documents = []
        self.doc_vectors = []
        self.terms = set()
        self.term_doc_freq = defaultdict(lambda: defaultdict(int))
        self.idf = {}
        self.tf_idf_vectors = []

    def add_document(self, doc_id, content):
        stemmed_text, _ = self.stemmer.stem_text(content)
        tokens = stemmed_text.split()
        
        self.documents.append({
            'id': doc_id,
            'content': content,
            'stemmed': stemmed_text,
            'tokens': tokens
        })
        
        for term in tokens:
            self.terms.add(term)
            self.term_doc_freq[term][doc_id] += 1

    def calculate_idf(self):
        N = len(self.documents)
        for term in self.terms:
            df = sum(1 for doc_id in range(N) if self.term_doc_freq[term][doc_id] > 0)
            self.idf[term] = math.log10((N + 1) / (df + 1)) + 1

    def calculate_weights(self):
        self.calculate_idf()
        self.doc_vectors = []
        self.tf_idf_vectors = []
        
        for doc in self.documents:
            vector = {}
            tf_idf_vector = {}
            doc_id = doc['id']
            
            for term in self.terms:
                tf = self.term_doc_freq[term][doc_id]
                if tf > 0:
                    weighted_tf = tf
                    vector[term] = tf
                    tf_idf_vector[term] = weighted_tf * self.idf[term]
                    
            self.doc_vectors.append(vector)
            self.tf_idf_vectors.append(tf_idf_vector)

    def search(self, query):
        stemmed_query, _ = self.stemmer.stem_text(query)
        query_terms = stemmed_query.split()

        query_vector = {}
        query_tf_idf = {}
        for term in query_terms:
            if term in self.terms:
                tf = query_terms.count(term)
                query_vector[term] = tf
                query_tf_idf[term] = tf * self.idf.get(term, 0)

        results = []
        for i, doc_vector in enumerate(self.tf_idf_vectors):
            similarity = self.cosine_similarity(query_tf_idf, doc_vector)
            results.append((self.documents[i], similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def cosine_similarity(self, vec1, vec2):
        if not vec1 or not vec2:
            return 0
            
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0) for term in self.terms)
        
        norm1 = math.sqrt(sum(value * value for value in vec1.values()))
        norm2 = math.sqrt(sum(value * value for value in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0
            
        return dot_product / (norm1 * norm2)
