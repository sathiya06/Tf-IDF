"""
    tm.py : Calculating Tf-IDF.    
    (C) 2024 Sathiya Narayanan Venkatesan (sathiyavenkat06@gmail.com) BSD-2 license.        
"""

import csv, math, re
from collections import defaultdict
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


file = "data/Hall.csv "

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
    # print(tokenized_documents[0])
    #stemming the tokens
    ps = PorterStemmer()
    punctuation = re.compile(r'[-.?!,:;()|0-9]')
    Stemmed_documents = [ [ ps.stem(token) for token in tokens if len(punctuation.sub("", token)) > 0] for tokens in tokenized_documents]
    # print("---------------------------------------------------------------------------------------")
    # print(Stemmed_documents[0])
    #Remove stop words
    stopWords = set(stopwords.words("english"))
    processed_documents = [ [ token for token in Stemmed_document if token not in stopWords ] for Stemmed_document in Stemmed_documents]
    # print("---------------------------------------------------------------------------------------")
    # print(processed_documents[0])
    #constructing ngrams fromm the tokens
    final_documents = [ [ ' '.join(ngram) for ngram in ngrams(processed_document, 4)] for processed_document in processed_documents]
    # print("---------------------------------------------------------------------------------------")
    # print(final_documents[0])
    return final_documents

def compute_tf1(doc):
    fdist = FreqDist()
    for n_gram in doc:
        fdist[n_gram.lower()] += 1
    return fdist

def compute_tf(doc):
    tf_dict = defaultdict(int)
    for word in doc:
        tf_dict[word] += 1
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / len(doc)
    return tf_dict

def compute_idf1(doc_list):

    fdist = FreqDist()
    N = len(doc_list)

    for doc in doc_list:
        for word in set(doc):
            fdist[word] += 1

    for word in fdist:
        fdist[word] = math.log(N / (fdist[word]))
    
    return fdist

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



documents = extract_text(file)
tokenized_documents = process_docs(documents)
# tokenized_documents = [word_tokenize(doc.lower()) for doc in documents]
tf_documents = [compute_tf(doc) for doc in tokenized_documents]
idf = compute_idf(tokenized_documents)
tfidf_documents = [compute_tfidf(tf_doc, idf) for tf_doc in tf_documents]
for i, doc in enumerate(tfidf_documents):
    print(f"Document {i+1} TF-IDF:")
    for word, score in doc.items():
        print(f"  {word}: {score}")
    if i == 2:
        break



