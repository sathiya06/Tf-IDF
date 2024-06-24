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

def extract_text(csv_file):
    titles_abstracts = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            titles_abstracts.append(row['Document Title'] + ' ' + row['Abstract'])
    return titles_abstracts

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

def compute_idf1(doc_list):
    tf_arr = []
    idf_dict = defaultdict(int)
    tf_fre = defaultdict(int)
    number_of_documents = len(doc_list)
    # number_of_tokens = 0

    # Count the number of documents that contain each word
    for doc in doc_list:
        # number_of_tokens += len(doc)
        for word in set(doc):
            idf_dict[word] += 1   

    for doc in doc_list:
        tf_dict = defaultdict(int)
        for word in doc:
            tf_dict[word] += 1
            tf_fre[word] += 1
        tf_arr.append(tf_dict)
    
    # Calculate the IDF score
    for word in idf_dict:
        idf_dict[word] = math.log(number_of_documents / (idf_dict[word]))
    
    for i, doc in enumerate(doc_list):
        tf_dict = tf_arr[i]
        for word in set(doc):
            tf_dict[word] = tf_dict[word] / tf_fre[word]
    
    return tf_arr, idf_dict

def compute_tfidf(tf_doc, idf):
    tfidf = {}
    for word, tf_val in tf_doc.items():
        tfidf[word] = tf_val * idf[word]
    return tfidf

def tokenize():
    documents = extract_text(file)
    tokenized_documents = process_docs(documents)
    return tokenized_documents

def one_for_all(tokenized_documents):
    # changed tf-idf
    tf_documents, idf = compute_idf1(tokenized_documents)
    tfidf_documents = [compute_tfidf(tf_doc, idf) for tf_doc in tf_documents]
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

def diff_for_all(tokenized_documents):
    tf_documents = [compute_tf(doc) for doc in tokenized_documents]
    idf = compute_idf(tokenized_documents)
    tfidf_documents = [compute_tfidf(tf_doc, idf) for tf_doc in tf_documents]
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

def print_fre(doc_list):

    tf_arr = []
    idf_dict = defaultdict(int)
    tf_fre = defaultdict(int)
    number_of_documents = len(doc_list)
    number_of_tokens = 0

    # Count the number of documents that contain each word
    for doc in doc_list:
        number_of_tokens += len(doc)
        for word in set(doc):
            idf_dict[word] += 1   

    for doc in doc_list:
        tf_dict = defaultdict(int)
        for word in doc:
            tf_dict[word] += 1
            tf_fre[word] += 1
        tf_arr.append(tf_dict)

    file_path = f'data/Hall/fre.csv'

    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['number of documents', number_of_documents])
        writer.writerow(['number_of_tokens', number_of_tokens])
        writer.writerow(['document', 'word', 'frequency'])
        # for key, value in tf_fre.items():
        #     writer.writerow(['all', key, value])
        for i, doc in enumerate(doc_list):
            writer.writerow([i, 'length', len(doc)])
            for key, val in tf_arr[i].items():
                writer.writerow([i, key, val])

def main():
    tokenized_documents = tokenize()

    print_fre(tokenized_documents)

    # diff_for_all(tokenized_documents)

    # one_for_all(tokenized_documents)

file = "data/Hall.csv"
as_words = False
nGram_size = 4

main()
