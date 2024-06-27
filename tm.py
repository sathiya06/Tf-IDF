"""
    tm.py : Calculating Tf-IDF.    
    (C) 2024 Sathiya Narayanan Venkatesan (sathiyavenkat06@gmail.com) BSD-2 license.        
"""

import csv, math, re, os
from collections import defaultdict
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def extract_text():
    titles_abstracts = []
    lables = []
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            titles_abstracts.append(row['Document Title'] + ' ' + row['Abstract'])
            lables.append(row['label'])
    return titles_abstracts, lables

def process_docs(documents):

    #Tokenizing the sentences
    tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]

    #stemming the tokens
    ps = PorterStemmer()
    punctuation = re.compile(r'[-.?!,:;()|0-9]')
    Stemmed_documents = [ [ ps.stem(token) for token in tokens if len(punctuation.sub("", token)) > 0] for tokens in tokenized_documents]

    #Remove stop words
    stop_words = set(stopwords.words("english"))
    processed_documents = [ [ token for token in Stemmed_document if token not in stop_words ] for Stemmed_document in Stemmed_documents]

    #constructing ngrams fromm the tokens words
    if as_words:
        final_documents = [ [ ' '.join(ngram) for ngram in ngrams(processed_document, nGram_size)] for processed_document in processed_documents]

    #constructing ngrams fromm the tokens characters
    else:
        character_tokens = [ [ char for token in processed_document for char in token ] for processed_document in processed_documents]
        final_documents = [ [ ' '.join(ngram) for ngram in ngrams(character_token, nGram_size)] for character_token in character_tokens]

    return final_documents

def compute_tf(doc):
    tf_dict = defaultdict(int)
    for word in doc:
        tf_dict[word] += 1
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / len(doc)
    return tf_dict

def compute_idf(doc_list):
    idf_dict = defaultdict(int)
    N = len(doc_list)
    
    # Count the number of documents that contain each word
    for doc in doc_list:
        for word in set(doc):
            idf_dict[word] += 1
    
    # Calculate the IDF score
    for word in idf_dict:
        idf_dict[word] = math.log(N / (idf_dict[word]))
    
    return idf_dict

def compute_tfidf(tf_doc, idf):
    tfidf = {}
    for word, tf_val in tf_doc.items():
        tfidf[word] = tf_val * idf[word]
    return tfidf

def tokenize():
    documents, labels = extract_text()
    tokenized_documents = process_docs(documents)
    return tokenized_documents, labels

def unsorted_print(tfidf_documents):
    for i, doc in enumerate(tfidf_documents):
        print(f"Document {i+1} TF-IDF:")
        j = 0
        doc = dict(sorted(doc.items(), key=lambda x: x[1], reverse=True))
        print(len(doc))
        for word, score in doc.items():
            print(f"{word}: {score}")
            j+=1
            if j == 50:
                break
        if i == 2:
            break

def get_top100(tfidf_documents):
    combined_tfidf = {}
    for doc in tfidf_documents:
        combined_tfidf.update(doc)
    sorted_tfidf = dict(sorted(doc.items(), key=lambda x: x[1], reverse=True))
    nGrams = []
    print('Top 100 Ngrams in all documents.')
    for word, score in sorted_tfidf.items():
        print(f"{word}: {score}")
        nGrams.append(word)
        if len(nGrams) == 100:
            break
    return nGrams

def generate_tabular_data(tfidf_documents, nGrams, labels):

    directory = os.path.dirname(output_file)
    os.makedirs(directory, exist_ok=True)

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        nGrams.append('label')
        writer.writerow(nGrams)
        
        for i, doc in enumerate(tfidf_documents):
            row = []
            for nGram in nGrams:
                row.append(round(doc.get(nGram, 0), 4))
            row.append(labels[i])
            writer.writerow(row)

def main():
    tokenized_documents, labels = tokenize()
    tf_documents = [compute_tf(doc) for doc in tokenized_documents]
    idf = compute_idf(tokenized_documents)
    tfidf_documents = [compute_tfidf(tf_doc, idf) for tf_doc in tf_documents]
    nGrams = get_top100(tfidf_documents)
    generate_tabular_data(tfidf_documents, nGrams, labels)

input_file = "data/Hall.csv"
output_file = f'data/new/Hall_words4.csv'
as_words = True
nGram_size = 4

main()
