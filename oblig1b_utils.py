import urllib.request
import pandas as pd
import sklearn.model_selection
import os
import re
import random

# METODER FOR LOGISTISK REGRESJON


ORDFILER = {
    "norsk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/nb.txt?raw=true",
    "arabisk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ar.txt?raw=true",
    "finsk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fi.txt?raw=true",
    "patwa": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/jam.txt?raw=true",
    "farsi": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fa.txt?raw=true",
    "tysk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/de.txt?raw=true",
    "engelsk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_UK.txt?raw=true",
    "rumensk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ro.txt?raw=true",
    "khmer": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/km.txt?raw=true",
    "fransk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/fr_FR.txt?raw=true",
    "japansk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ja.txt?raw=true",
    "spansk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/es_ES.txt?raw=true",
    "svensk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/sv.txt?raw=true",
    "koreansk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ko.txt?raw=true",
    "swahilisk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/sw.txt?raw=true",
    "vietnamesisk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/vi_C.txt?raw=true",
    "mandarin": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/zh_hans.txt?raw=true",
    "malayisk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/ms.txt?raw=true",   # corrected from ma.txt
    "kantonesisk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/yue.txt?raw=true",
    "islandsk": "https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/is.txt?raw=true"
}


def extract_wordlist(cache_file="./langid_data.csv"):
    """Returnerer 2 Dataframes (en for trening og en for testing)."""
    
    if cache_file is not None and os.path.exists(cache_file):
        print(f"Reading cached file from {cache_file}")
        # FIXED: Use python engine to avoid ParserError with IPA characters
        full_wordlist = pd.read_csv(
            cache_file,
            engine='python',
            encoding='utf-8',
            on_bad_lines='skip'   # skip any problematic lines
        )
    else:
        print("No cache found. Downloading data from GitHub...")
        full_wordlist = _download_wordlist()
        if cache_file is not None:
            full_wordlist.to_csv(cache_file, index=False, encoding='utf-8')
            print(f"Data saved to cache: {cache_file}")
            
    # Split into train and test (10% test)
    train, test = sklearn.model_selection.train_test_split(
        full_wordlist, test_size=0.1, random_state=42
    )
        
    print(f"Treningsett: {len(train):,} eksempler, testsett: {len(test):,} eksempler")
    return train, test


def _download_wordlist(max_nb_words_per_language=50000):
    """Laster ned ordlister fra GitHub og bygger en stor DataFrame."""
    
    full_wordlist = []
    
    for lang, wordfile in ORDFILER.items():
        print(f"Nedlasting av ordlisten for {lang}... ", end="")
        
        try:
            with urllib.request.urlopen(wordfile) as response:
                data = response.read().decode("utf-8")
            
            wordlist_for_language = []
            for linje in data.splitlines():
                linje = linje.strip()
                if not linje or "\t" not in linje:
                    continue
                    
                parts = linje.split("\t", 1)  # split only on first tab
                if len(parts) != 2:
                    continue
                    
                word, transcription = parts
                
                # Fix primary stress symbol
                transcription = transcription.replace("'", "ˈ")
                
                # Extract the first transcription inside /.../
                match = re.search(r"/(.+?)/", transcription)
                if not match:
                    continue
                    
                transcription = match.group(1).strip()
                
                wordlist_for_language.append({
                    "ord": word,
                    "IPA": transcription,
                    "språk": lang
                })
            
            # Shuffle and limit size
            random.shuffle(wordlist_for_language)
            wordlist_for_language = wordlist_for_language[:max_nb_words_per_language]
            
            full_wordlist.extend(wordlist_for_language)
            print("ferdig!")
            
        except Exception as e:
            print(f"FEIL ved nedlasting av {lang}: {e}")
    
    # Build DataFrame
    full_wordlist = pd.DataFrame.from_records(full_wordlist)
    
    # Shuffle all data
    full_wordlist = full_wordlist.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return full_wordlist

# METODER FOR ENTITETSGJENKJENNING


def preprocess(tagged_text):
    """Tar en tokenisert tekst med XML tags og returnerer setninger + spans."""
    
    sentences = []
    spans = []
    
    for line in tagged_text.split("\n"):
        tokens = []
        spans_in_sentence = []
        
        for j, token in enumerate(line.split(" ")):
            if not token:
                continue
                
            # Opening tag
            start_match = re.match(r"<(\w+?)>", token)
            if start_match:
                new_span = [j, None, start_match.group(1)]
                spans_in_sentence.append(new_span)
                token = token[start_match.end(0):]
            
            # Closing tag
            end_match = re.match(r"(.+)</(\w+?)>$", token)
            if end_match:
                if not spans_in_sentence or spans_in_sentence[-1][1] is not None:
                    continue  # skip malformed
                start, _, tag = spans_in_sentence[-1]
                if tag != end_match.group(2):
                    continue
                token = end_match.group(1)
                spans_in_sentence[-1][1] = j + 1
            
            tokens.append(token)
        
        sentences.append(tokens)
        spans.append(spans_in_sentence)
        
    return sentences, spans


def get_spans(label_sequence):
    """Convert BIO labels to spans."""
    spans = []
    i = 0
    while i < len(label_sequence):
        label = label_sequence[i]
        if label.startswith("B-"):
            start = i
            tag = label[2:]
            end = start + 1
            while end < len(label_sequence) and label_sequence[end] == f"I-{tag}":
                end += 1
            spans.append((start, end, tag))
            i = end
        else:
            i += 1
    return spans


def postprocess(sentences, spans):
    """Convert sentences and spans back to XML-tagged text."""
    tagged_sentences = []
    
    for sentence, sentence_spans in zip(sentences, spans):
        new_sentence = list(sentence)
        for start, end, tag in sentence_spans:
            if start < len(new_sentence):
                new_sentence[start] = f"<{tag}>{new_sentence[start]}"
            if end - 1 < len(new_sentence):
                new_sentence[end - 1] = f"{new_sentence[end - 1]}</{tag}>"
        tagged_sentences.append(" ".join(new_sentence))
    
    return "\n".join(tagged_sentences)
