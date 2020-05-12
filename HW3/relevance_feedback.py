# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    rf_sim = sim
    vec_queries = vec_queries.toarray()
    vec_docs = vec_docs.toarray()
    
    no_queries = sim.shape[1]  #no. of queries
    no_docs = sim.shape[0]     #no.of documents
    
    beta = 0.75  #beta
    gamma = 0.25  #gamma
    
    normalised_beta = beta/n
    normalise_gamma = gamma/n
    
    no_iterations = 3
    iteration = 0
    while iteration < no_iterations:            
        for i in range(no_queries):
            sorted_docs = np.argsort(-rf_sim[:, i])    #in descending order of relevance
            rel = sorted_docs[:n]
            non_rel = sorted_docs[-n:]
            
            temp_quer = vec_queries[i] + normalised_beta*np.sum(vec_docs[rel], axis = 0) - normalise_gamma*np.sum(vec_docs[non_rel], axis = 0)
            
            vec_queries[i] = temp_quer         
            
        rf_sim = cosine_similarity(vec_docs, vec_queries)
        iteration += 1
        
     # change
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant
        k: integer
            number of words to be added to the query

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    
    beta = 0.6
    gamma = 0.4
    
    normalised_beta = beta/n
    normalise_gamma = gamma/n
    
    vec_queries = vec_queries.toarray()
    vec_docs = vec_docs.toarray()
    
    no_queries = sim.shape[1]  #no. of queries
    no_docs = sim.shape[0]     #no.of documents
    
    no_iterations = 3
    iteration = 0
    
    k = 5 #no. of words added to a query per iteration
    rf_sim = sim
    
    non_zero = []
    for i in range(no_queries):
        quer = []
        for j in range(vec_queries.shape[1]):
            if vec_queries[i][j] != 0:
                quer.append(j)
        non_zero.append(quer)
    
    while iteration < no_iterations:
        q = 0
        while q < no_queries:
            sorted_docs = np.argsort(-rf_sim[:, q])    #in descending order of relevance
            rel = sorted_docs[:n]
            non_rel = sorted_docs[-n:]
            
            sorted_words = np.argsort(-np.sum(vec_docs[rel], axis = 0))
            
            new_words = []
            words_added = 0
            
            idx = 0  #to loop over sorted_words
            while words_added != k:
                if sorted_words[idx] not in non_zero[q]:  #if the word isn't present in original query
                    new_words.append(sorted_words[idx])
                    words_added += 1
                    
                idx += 1
            
            new_words.extend(non_zero[q])  #all the words in the updated query
            non_zero[q] = new_words
            
            table = tfidf_model.get_feature_names()   
            all_words_terms = [table[key] for key in new_words]  #the actual words corresponding to the indices
            
            query_string = ' '.join(all_words_terms)  #string of the quer reqd for .transform
            new_query = tfidf_model.transform([query_string])
            new_query = new_query.toarray()[0]
            
            vec_queries[q] = new_query
            
            for idx in range(no_docs):
                if idx not in non_zero[q]:
                    vec_queries[q][idx] = 0
                    
            #Vector Adjustment
            temp_quer = vec_queries[q] + normalised_beta*np.sum(vec_docs[rel], axis = 0) - normalise_gamma*np.sum(vec_docs[non_rel], axis = 0)
            vec_queries[q] = temp_quer  
            
            q+=1
            
        rf_sim = cosine_similarity(vec_docs, vec_queries)
        iteration+=1
                
    rf_sim = cosine_similarity(vec_docs, vec_queries)
    #rf_sim = sim  # change
    return rf_sim