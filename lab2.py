import math

class NaiveBayes:
    def __init__(self):
        self.classes = {}
        self.vocab = set()
        self.class_word_counts = {}
        self.class_counts = {}
        self.total_words = {}

    def train(self, data, labels):        
        for c in set(labels):
            self.class_counts[c] = labels.count(c)
            self.class_word_counts[c] = {}
            self.total_words[c] = 0
               
        for text, label in zip(data, labels):
            words = text.split()
            for word in words:
                self.vocab.add(word)
                self.class_word_counts[label][word] = self.class_word_counts[label].get(word, 0) + 1
                self.total_words[label] += 1
    
    def predict(self, text):
        words = text.split()
        scores = {}
        total_docs = sum(self.class_counts.values())
        
        for c in self.class_counts:           
            scores[c] = math.log(self.class_counts[c] / total_docs)
            
            for word in words:
                word_count = self.class_word_counts[c].get(word, 0) + 1
                word_prob = word_count / (self.total_words[c] + len(self.vocab))
                scores[c] += math.log(word_prob)
        
        return max(scores, key=scores.get)

# Sample dataset
data = [
    "buy cheap medicine",
    "cheap price discount",
    "medicine for sale",
    "meeting at noon",
    "schedule the appointment",
    "lunch at office"
]
labels = ["spam", "spam", "spam", "ham", "ham", "ham"]

nb = NaiveBayes()
nb.train(data, labels)

test_text = "cheap discount offer"
print("Prediction:", nb.predict(test_text))
