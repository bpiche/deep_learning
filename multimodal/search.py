import glob
import spacy
import tqdm
import json
import argparse
import numpy as np
import pandas as pd


# an argument for the user query input
parser = argparse.ArgumentParser()
parser.add_argument('--query', type=str, default='Show me a paper about the LayoutLMv3 model.')
args = parser.parse_args()

tqdm.tqdm.pandas()

nlp = spacy.load('en_core_web_lg', 
                 exclude=["tagger", 
                          "parser", 
                          "senter", 
                          "attribute_ruler", 
                          "lemmatizer", 
                          "ner"])


def vectorize_docs():
    """
    """
    l = []
    for f in glob.glob('./data/*.json'):
        print(f)
        d = pd.read_json(f)
        d = d[['Title', 'Authors', 'Abstract', 'Key Findings']]
        # just append the first row
        l.append(d.iloc[0])
    df = pd.concat(l, axis=1).T
    # vectorize the docs
    df['sent_vectors'] = df.Abstract.progress_apply(lambda x: nlp(x).vector)
    return df


def compare_docs(query, df):
    """
    """
    # compare vectorized user input to vectorized documents
    query_vect = nlp(query).vector
    df['similarity'] = df.sent_vectors.apply(lambda x: np.dot(x, query_vect) / (np.linalg.norm(x) * np.linalg.norm(query_vect)))
    results_df = df.sort_values('similarity', ascending=False)[0:10]
    # return a response with the top 10 results from the results_df dataframe
    response = {'results': []}
    for index, row in results_df.iterrows():
        response['results'].append({'title': row['Title'], 
                                    'authors': row['Authors'],
                                    # 'abstract': row['Abstract'],
                                    # 'key_findings': row['Key Findings'],
                                    'similarity': row['similarity']})
    return response


if __name__ == "__main__":
    """
    """
    df = vectorize_docs()
    query = args.query
    response = compare_docs(query, df)
    with open('./data/response.json', 'w') as f:
        json.dump(response, f)