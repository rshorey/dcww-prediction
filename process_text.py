import string
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler



def clean_text(filename):
    bad_chars = string.punctuation + string.digits + string.whitespace.replace(" ","")
    trans_table = str.maketrans("","",bad_chars) #this is the python 3 way of removing chars w translate
    #translate is somewhat faster than looping and doing replace, I think
    sentences = []
    with open(filename,'r') as f:
        for line in f:
            lines = line.split(".")
            for s in lines:
                s = s.translate(trans_table)
                s = s.lower()
                s = s.replace("article","")
                s = s.replace("artículo","")
                s = s.strip()
                if len(s) > 0:
                    sentences.append(s)
    return sentences
    
def make_features(sentences,class_name,sentence_features=[],sentence_ids={}):
    #takes a list of sentences and makes the features we care about.
    #Returns a list of sentence_id, class, features for each sentence
    #and a dictrionary from ids to sentences
    
    feature_def = ["th","la","el"," y ","and","x","sh","k","ó","í"," ll","gh"," st"]
    #will check if each feature appears in the sentence and add a 0 or 1 to use as a DV
    #these were specifically thrown in as an example for english/spanish.
    #if you want to try other languages, probably you should define more features
    #OR write code to check every letter or letter pair!
    
    
    sentence_id = 0
    if len(sentence_ids) > 0:
        sentence_id = max(sentence_ids.keys())+1
    
    #we will also include average word length
    for s in sentences:
        sentence_ids[sentence_id] = s #add sentence to our dictionary
        sentence_feat = [sentence_id,class_name] #we'll need to know which sentence and which class
        sentence_id += 1
        
        #append the features we defined above
        for f in feature_def:
            if f in s:
                sentence_feat.append(1)
            else:
                sentence_feat.append(0)
                
        #also append the average word length, that might be an interesting feature
        words = s.split(" ")
        word_lengths = 0
        for w in words:
            word_lengths += len(w)
        sentence_feat.append(word_lengths/len(words))
        
        sentence_features.append(sentence_feat)
        
    col_names = ["index","language"]
    for f in feature_def:
        col_names.append(f.replace(" ","^"))
    col_names.append("avg_word_len")
    return sentence_features, sentence_ids, col_names


def create_lr_model(list_of_data, col_names):
    df = pd.DataFrame(list_of_data,columns=col_names)
    df.set_index('index',inplace = True)
    y = df.loc[:,"language"].values
    X = df.drop("language",axis=1).values
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(C=0.1, penalty='l1',tol=0.01)
    clf = lr.fit(X_train_scaled,y_train)
    y_pred = clf.predict(X_test_scaled)

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    coefs = clf.coef_[0]
    for i in range(len(coefs)):
        print("{0} : {1}".format(col_names[i+2],coefs[i]))
    
    return (scaler, clf)
    
eng = clean_text("english_raw.txt")
sp = clean_text("spanish_raw.txt")
features,sentence_dict,col_names = make_features(eng,0)
features,sentence_dict,col_names = make_features(sp,1,features,sentence_dict)

