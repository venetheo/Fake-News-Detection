import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import utils,preprocessing,feature_extraction,feature_selection, model_selection, naive_bayes, pipeline, manifold, preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from tensorflow.keras import models,layers
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import nltk
import re
import transformers
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import multiprocessing

def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words)) for doc in sents])
    return targets, regressors

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens

def runDoc2Vec(train,test,epochs):
    train['title']=train['title'].apply(cleanText)
    test['title']=test['title'].apply(cleanText)
    gtraintagged=train.apply(lambda r: TaggedDocument (words=tokenize_text(r['title']),
                                                          tags=[r.Label]),axis=1)
    gtesttagged=test.apply(lambda r: TaggedDocument (words=tokenize_text(r['title']),
                                                        tags=[r.Label]),axis=1)
    cores = multiprocessing.cpu_count()
    #dbow
    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores)
    model_dbow.build_vocab([x for x in tqdm(gtraintagged.values)])
    for epoch in range(epochs):
        model_dbow.train(utils.shuffle([x for x in tqdm(gtraintagged.values)]), total_examples=len(gtraintagged.values), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha
    y_train, X_train = vec_for_learning(model_dbow, gtraintagged)
    y_test_dbow, X_test = vec_for_learning(model_dbow, gtesttagged)
    pipedbow=make_pipeline(StandardScaler(), LogisticRegression(n_jobs=1, C=1e5))
    pipedbow.fit(X_train, y_train)
    y_pred_dbow = pipedbow.predict(X_test)
    #dm
    model_dmm=Doc2Vec(dm=1, dm_mean=1, vector_size=300, window=10, negative=5, min_count=1, workers=5, alpha=0.065, min_alpha=0.065)
    model_dmm.build_vocab([x for x in tqdm(gtraintagged.values)])
    for epoch in range(epochs):
        model_dmm.train(utils.shuffle([x for x in tqdm(gtraintagged.values)]), total_examples=len(gtraintagged.values), epochs=1);
        model_dmm.alpha -= 0.002
        model_dmm.min_alpha = model_dmm.alpha
    y_train, X_train = vec_for_learning(model_dbow, gtraintagged)
    y_test_dm, X_test = vec_for_learning(model_dbow, gtesttagged)
    pipedbow=make_pipeline(StandardScaler(), LogisticRegression(n_jobs=1, C=1e5))
    pipedbow.fit(X_train, y_train)
    y_pred_dm = pipedbow.predict(X_test)
    #combined
    #model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    #model_dmm.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    new_model = ConcatenatedDoc2Vec([model_dbow, model_dmm])
    y_train, X_train = vec_for_learning(new_model, gtraintagged)
    y_test_combined, X_test = vec_for_learning(new_model, gtesttagged)
    pipecomb=make_pipeline(StandardScaler(), LogisticRegression(n_jobs=1, C=1e5))
    pipecomb.fit(X_train, y_train)
    y_pred_combined = pipecomb.predict(X_test)
    return {
        'Accuracy Doc2Vec(DBOW)': accuracy_score(y_test_dbow, y_pred_dbow),
        'F1 Doc2Vec(DBOW)': f1_score(y_test_dbow, y_pred_dbow, average='weighted'),
        'Accuracy Doc2Vec(DM)': accuracy_score(y_test_dm, y_pred_dm),
        'F1 Doc2Vec(DM)': f1_score(y_test_dm, y_pred_dm, average='weighted'),
        'Accuracy Doc2Vec(Combined)': accuracy_score(y_test_combined, y_pred_combined),
        'F1 Doc2Vec(Combined)':f1_score(y_test_combined, y_pred_combined, average='weighted')}

def runtfidf(train,test):
    train['title']=train['title'].apply(cleanText)
    test['title']=test['title'].apply(cleanText)
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000,ngram_range=(1,2))
    corpus = train["title"]
    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_
    y = train["Label"]
    X_names = vectorizer.get_feature_names()
    p_value_limit = 0.95
    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X_train, y==cat)
        dtf_features = dtf_features.append(pd.DataFrame({"feature":X_names, "score":1-p, "y":cat}))
        dtf_features = dtf_features.sort_values(["y","score"],ascending=[True,False])
        dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
    X_names = dtf_features["feature"].unique().tolist()
    cf=LogisticRegression(n_jobs=1,C=1e5)
    pipe=pipeline.Pipeline([('vectorizer',vectorizer),('classifier',cf)])
    pipe['classifier'].fit(X_train,y.values)
    X_test=test['title'].values
    y_test=test['Label'].values
    pred=pipe.predict(X_test)
    return {'Accuracy Tf-Idf':accuracy_score(y_test,pred),'F1 Tf-Idf':f1_score(y_test,pred)}

def runbow(train,test):
    train['title']=train['title'].apply(cleanText)
    test['title']=test['title'].apply(cleanText)
    vectorizer = feature_extraction.text.CountVectorizer(max_features=10000,ngram_range=(1,2))
    corpus = train["title"]
    vectorizer.fit(corpus)
    X_train = vectorizer.transform(corpus)
    dic_vocabulary = vectorizer.vocabulary_
    y = train["Label"]
    X_names = vectorizer.get_feature_names()
    p_value_limit = 0.95
    dtf_features = pd.DataFrame()
    for cat in np.unique(y):
        chi2, p = feature_selection.chi2(X_train, y==cat)
        dtf_features = dtf_features.append(pd.DataFrame({"feature":X_names, "score":1-p, "y":cat}))
        dtf_features = dtf_features.sort_values(["y","score"],ascending=[True,False])
        dtf_features = dtf_features[dtf_features["score"]>p_value_limit]
    X_names = dtf_features["feature"].unique().tolist()
    cf=LogisticRegression(n_jobs=1,C=1e5)
    pipe=pipeline.Pipeline([('vectorizer',vectorizer),('classifier',cf)])
    pipe['classifier'].fit(X_train,y.values)
    X_test=test['title'].values
    y_test=test['Label'].values
    pred=pipe.predict(X_test)
    return {'Accuracy BOW':accuracy_score(y_test,pred),'F1 BOW':f1_score(y_test,pred)}

def featurizeforBert(input):
    corpus = input["title"].dropna()
    maxlen = 50
    tokenizer=transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    ## add special tokens
    maxqnans = int((maxlen-20)/2)
    corpus_tokenized = ["[CLS] "+" ".join(tokenizer.tokenize(re.sub(r'[^\w\s]+|\n', '',str(txt).lower().strip()))[:maxqnans])+" [SEP] " for txt in corpus]
    ## generate masks
    masks = [[1]*len(txt.split(" ")) + [0]*(maxlen - len(txt.split(" "))) for txt in corpus_tokenized]
    ## padding
    txt2seq = [txt + " [PAD]"*(maxlen-len(txt.split(" "))) if len(txt.split(" ")) != maxlen else txt for txt in corpus_tokenized]
    ## generate idx
    idx = [[tokenizer.encode(x) for x in seq.split(" ")]  for seq in txt2seq ]

    ## generate segments
    segments = [] 
    for seq in txt2seq:
        temp, i = [], 0
        for token in seq.split(" "):
            temp.append(i)
            if token == "[SEP]":
                 i += 1
        segments.append(temp)
    ## feature matrix
    return [np.asarray(idx, dtype='int32'), 
               np.asarray(masks, dtype='int32'), 
               np.asarray(segments, dtype='int32')]

def runBERT(train,test):
    
    X_train=featurizeforBert(train)
    y_train=train['Label'].values
    X_test=featurizeforBert(test)
    y_test=train['Label'].values
    ## inputs
    idx = layers.Input((50), dtype="int32", name="input_idx")
    masks = layers.Input((50), dtype="int32", name="input_masks")
    ## pre-trained bert with config
    config = transformers.DistilBertConfig(dropout=0.2,attention_dropout=0.2)
    config.output_hidden_states = False
    nlp = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
    bert_out = nlp(idx, attention_mask=masks)[0]
    ## fine-tuning
    x = layers.GlobalAveragePooling1D()(bert_out)
    x = layers.Dense(64, activation="relu")(x)
    y_out = layers.Dense(len(np.unique(y_train)), activation='softmax')(x)
    ## compile
    model = models.Model([idx, masks], y_out)
    for layer in model.layers[:3]:
        layer.trainable = False
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_train))}
    inverse_dic = {v:k for k,v in dic_y_mapping.items()}
    y_train = np.array([inverse_dic[y] for y in y_train])
    training = model.fit(x=X_train, y=y_train, batch_size=64,epochs=1, shuffle=True, verbose=1, validation_split=0.3)
    pred = model.predict(X_test)
    return {'Accuracy BERT':accuracy_score(y_test,pred),'F1 BERT':f1_score(y_test,pred)}

def runBERT2(data):
    X_train, X_test, y_train, y_test = train_test_split(data['title'],data['Label'],stratify=data['Label'])
    bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessed_text = bert_preprocess(text_input)
    outputs = bert_encoder(preprocessed_text)
    l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)
    model = tf.keras.Model(inputs=[text_input], outputs = [l])
    METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy')]
    model.compile(optimizer='adam',
     loss='binary_crossentropy',
     metrics=METRICS)
    model.fit(X_train,y_train,epochs=10)
    y_pred=model.predict(X_test)
    y_pred=y_pred.flatten()
    pred = np.where(y_predicted > 0.5, 1, 0)
    return {'Acc':accuracy_score(y_test,pred),'F1':f1_score(y_test,pred)}
    
    
#nltk.download("popular")
#initialize datasets
file_cols=['id','title','news_url']
gossipli=[]
gossipli.append(pd.read_csv('./FakeNewsNet/dataset/gossipcop_fake.csv',index_col=None,usecols=file_cols).assign(Label=False))
gossipli.append(pd.read_csv('./FakeNewsNet/dataset/gossipcop_real.csv',index_col=None,usecols=file_cols).assign(Label=True))
gossip=pd.concat(gossipli,axis=0,ignore_index=True)
politili=[]
politili.append(pd.read_csv('./FakeNewsNet/dataset/politifact_fake.csv',index_col=None,usecols=file_cols).assign(Label=False))
politili.append(pd.read_csv('./FakeNewsNet/dataset/politifact_real.csv',index_col=None,usecols=file_cols).assign(Label=True))
politi=pd.concat(politili,axis=0,ignore_index=True)
print(gossip.size)
print(politi.size)

#split dataset
gossip_train, gossip_test =train_test_split(gossip,test_size=0.3,shuffle=True)
politi_train, politi_test =train_test_split(politi,test_size=0.3,shuffle=True)

#runs
gBERTres=runBERT2(gossip)
gtfidfres=runtfidf(gossip_train,gossip_test)
gbowres=runbow(gossip_train,gossip_test)
gdoc2vecres=runDoc2Vec(gossip_train,gossip_test,30)

pBERTres=runBERT2(politi)
ptfidfres=runtfidf(politi_train,politi_test)
pbowres=runbow(politi_train,politi_test)
pdoc2vecres=runDoc2Vec(politi_train,politi_test,30)


##print results
print('----------=================Gossip Results==================----------------')
print('Results for Tf-Idf ',gtfidfres)
print('Results for BOW ',gbowres)
print('Results for Doc2Vec ',gdoc2vecres)
print('Results for BERT ',gBERTres)

print('----------=================Politi Results==================----------------')
print('Results for Tf-Idf ',ptfidfres)
print('Results for BOW ',pbowres)
print('Results for Doc2Vec ',pdoc2vecres)
print('Results for BERT ',pBERTres)


