import oblig1b_utils 
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

train_data, test_data = oblig1b_utils.extract_wordlist()

print("Statistikk over språkene i treningsett:")
print(train_data.språk.value_counts())
print("Første 30 ord:")
print(train_data[:30])

import sklearn.linear_model


class LanguageIdentifier:
    """Logistisk regresjonsmodell som tar IPA transkripsjoner av ord som input."""

    def __init__(self):
        self.model = sklearn.linear_model.LogisticRegression(solver="liblinear", multi_class='ovr')

    def train(self, transcriptions, languages):
        if len(transcriptions) != len(languages):
            raise ValueError("Lengden for transkripsjoner og språk er ikke lik.")
        
        X = self._extract_feats(transcriptions)
        
        unique_languages = list(set(languages))
        self.language_to_label = {lang: i for i, lang in enumerate(unique_languages)}

        y = [self.language_to_label[lang] for lang in languages]
        
        self.model.fit(X, y)


    def predict(self, transcriptions):
        X = self._extract_feats(transcriptions)
        predicted_labels = self.model.predict(X)
        
        predicted_languages = []
        for label in predicted_labels:
            for lang, lbl in self.language_to_label.items():
                if lbl == label:
                    predicted_languages.append(lang)
                    break
        return predicted_languages


    def _extract_unique_symbols(self, transcriptions, min_nb_occurrences=10):
        AntSymboler = {}
        for transcription in transcriptions:
            for symbol in transcription:
                AntSymboler[symbol] = AntSymboler.get(symbol, 0) + 1
        
        unique_symbols = [symbol for symbol, count in AntSymboler.items() 
                         if count >= min_nb_occurrences]
        return unique_symbols


    def _extract_feats(self, transcriptions):
        unique_symbols = self._extract_unique_symbols(train_data.IPA.values)
        m = len(unique_symbols)
        n = len(transcriptions)
        print("antall unike symboler:", m, "\n")

        X = np.zeros((n, m))
        symbol_to_index = {symbol: idx for idx, symbol in enumerate(unique_symbols)}

        for i, transcription in enumerate(transcriptions):
            for symbol in transcription:
                if symbol in symbol_to_index:
                    j = symbol_to_index[symbol]
                    X[i, j] = 1
        return X


    def evaluate(self, transcriptions, languages):
        predicted_languages = self.predict(transcriptions)
        
        accuracy = accuracy_score(languages, predicted_languages)
        print("Accuracy:", accuracy, "\n")
        
        precision = precision_score(languages, predicted_languages, average=None, zero_division=0)
        print("precision:", precision, "\n")
        
        recall = recall_score(languages, predicted_languages, average=None, zero_division=0)
        print("recall:", recall, "\n")
        
        f1_scores = f1_score(languages, predicted_languages, average=None, zero_division=0)
        print("F1 Score:", f1_scores, "\n")
        
        f1_micro = f1_score(languages, predicted_languages, average='micro')
        print("f1_micro:", f1_micro)
        
        f1_macro = f1_score(languages, predicted_languages, average='macro')
        print("f1_macro:", f1_macro)


# Bind the methods (this must come AFTER the function definitions)
LanguageIdentifier._extract_unique_symbols = LanguageIdentifier._extract_unique_symbols
LanguageIdentifier._extract_feats = LanguageIdentifier._extract_feats
LanguageIdentifier.train = LanguageIdentifier.train
LanguageIdentifier.predict = LanguageIdentifier.predict
LanguageIdentifier.evaluate = LanguageIdentifier.evaluate


# ====================== TRAIN & TEST LANGUAGE IDENTIFIER ======================
model = LanguageIdentifier()
model.train(train_data.IPA.values, train_data.språk.values)

predicted_langs = model.predict(["konstituˈθjon", "ɡrʉnlɔʋ", "stjourtnar̥skrauːɪn", "perusˌtuslɑki"])
print("Mest sansynnlige språk for ordene:", predicted_langs)

print("\n=== EVALUERING PÅ TESTSETT ===")
model.evaluate(test_data.IPA.values, test_data.språk.values)

# Most important symbol for Norwegian
norsk_index = model.language_to_label["norsk"]
weights_norsk = model.model.coef_[norsk_index]
max_weight_index = weights_norsk.argmax()

unique_symbols = model._extract_unique_symbols(train_data.IPA.values)
symbolet = unique_symbols[max_weight_index]
print("\nDet fonetiske symbolet som bidrar mest til norsk er:", symbolet)


# ====================== NAMED ENTITY RECOGNISER ======================
print("\nTesting preprocess...")
oblig1b_utils.preprocess("De første 43 minuttene hadde <ORG>Rosenborg</ORG> all makt og " +
                        "tilnærmet full kontroll på <LOC>Fredrikstad Stadion</LOC> .")


class NamedEntityRecogniser:
    def __init__(self):
        self.labels = set()
        self.vocab = set()
        self.label_counts = {}
        self.transition_counts = {}
        self.emission_counts = {("O", "<UNK>"): 1}
        self.transition_probs = {}
        self.emission_probs = {}

    def fit(self, tagged_text):
        sentences, all_spans = oblig1b_utils.preprocess(tagged_text)
        for sentence, spans in zip(sentences, all_spans):
            label_sequence = self.get_BIO_sequence(spans, len(sentence))
            self._add_counts(sentence, label_sequence)
        self._fill_probs()

    def _add_counts(self, sentence, label_sequence):
        if len(sentence) != len(label_sequence):
            raise ValueError("ikke samme lengde på sentence og label_sequence")
        
        self.vocab.update(sentence)
        self.labels.update(label_sequence)
        
        for label in label_sequence:
            self.label_counts[label] = self.label_counts.get(label, 0) + 1
        
        for i in range(len(sentence)):
            word = sentence[i]
            label = label_sequence[i]
            self.emission_counts[(label, word)] = self.emission_counts.get((label, word), 0) + 1
        
        for label1, label2 in zip(['<s>'] + label_sequence, label_sequence + ['</s>']):
            self.transition_counts[(label1, label2)] = self.transition_counts.get((label1, label2), 0) + 1

    def get_BIO_sequence(self, spans, sentence_length):
        bio_sequence = ['O'] * sentence_length
        for start, end, tag in spans:
            start = max(0, min(start, sentence_length))
            end = max(0, min(end, sentence_length))
            bio_sequence[start] = 'B-' + tag
            for i in range(start + 1, end):
                bio_sequence[i] = 'I-' + tag
        return bio_sequence

    def _fill_probs(self, alpha_smoothing=1E-6):
        # Transition probabilities
        for prev_label in self.labels:
            self.transition_probs[prev_label] = {}
            total = sum(self.transition_counts.get((prev_label, lbl), 0) for lbl in self.labels)
            for next_label in self.labels:
                count = self.transition_counts.get((prev_label, next_label), 0)
                self.transition_probs[prev_label][next_label] = (count + alpha_smoothing) / (total + alpha_smoothing * len(self.labels))
        
        # Emission probabilities with Laplace smoothing
        for label in self.labels:
            self.emission_probs[label] = {}
            label_count = self.label_counts.get(label, 0)
            for token in self.vocab:
                count = self.emission_counts.get((label, token), 0)
                self.emission_probs[label][token] = (count + alpha_smoothing) / (label_count + alpha_smoothing * len(self.vocab))

    def _beam_search(self, sentence, beam_width=3):
        beam = [{"labels": [], "prob": 1.0}]

        for token in sentence:
            new_beam = []
            for hypo in beam:
                prev_label = '<s>' if not hypo["labels"] else hypo["labels"][-1]
                
                for label in self.labels:
                    new_labels = hypo["labels"] + [label]
                    emit_prob = self.emission_probs.get(label, {}).get(token, 0)
                    trans_prob = self.transition_probs.get(prev_label, {}).get(label, 0)
                    new_prob = hypo["prob"] * emit_prob * trans_prob
                    new_beam.append({"labels": new_labels, "prob": new_prob})
            
            new_beam.sort(key=lambda x: x["prob"], reverse=True)
            beam = new_beam[:beam_width]

        best = max(beam, key=lambda x: x["prob"])
        return best["labels"], best["prob"]

    def label(self, text):
        sentences, _ = oblig1b_utils.preprocess(text)
        spans_list = []
        for sentence in sentences:
            sentence = [tok if tok in self.vocab else "<UNK>" for tok in sentence]
            label_seq, _ = self._beam_search(sentence)
            spans_list.append(oblig1b_utils.get_spans(label_seq))
        return oblig1b_utils.postprocess(sentences, spans_list)


# Bind the methods for NER
NamedEntityRecogniser._add_counts = NamedEntityRecogniser._add_counts
NamedEntityRecogniser._fill_probs = NamedEntityRecogniser._fill_probs
NamedEntityRecogniser._beam_search = NamedEntityRecogniser._beam_search


# Train NER model
with open("norne_train.txt", encoding="utf-8") as fd:
    training_text = fd.read()

ner_model = NamedEntityRecogniser()
ner_model.fit(training_text)

# Test
print("\nTesting NER:")
print(ner_model.label("Kjell Magne Bondevik var statsminister i Norge ."))