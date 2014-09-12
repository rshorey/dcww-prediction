import string
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import csv



def clean_text(filename):
    bad_chars = string.punctuation + string.digits + string.whitespace.replace(" ","")+"«»"
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
    #to start fresh, exclude sentence_features and sentence_ids
    #to add to an existing corpus, pass the values you've already created
    
    feature_def = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o",
                    "p","q","r","s","t","u","v","w","x","y","z"]
    #will check if each feature appears in the sentence and add a 0 or 1 to use as a DV
    
    
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


def create_lr_model(list_of_data, col_names, index_name, dv_name):
    #1)list_of_data is a list of lists. Each list contains the following values, in this order:
    #a) the id of the thing we're predicting.
    #b) the class, or dependent variable (a 1 or 0)
    #c) all of the independent varaibles we'll use to predict
    
    #2) col_names is the header column for list_of_data
    
    #3) index_name is the name of the column we'll be using as the index
    
    #4) dv_name is the name of the dependent varaible or class we'll be predicting
    
    #prep the data for modeling
    df = pd.DataFrame(list_of_data,columns=col_names) #turn the list into a pandas data frame
    df.set_index(index_name,inplace = True) #create an index on the dataframe using the sentence id
    y = df.loc[:,dv_name] #pulls out the dependent variable

    X = df.drop(dv_name,axis=1) #removes the dependent variable from the independent variables

    #split into training and test sets
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    
    #scale the data to deal with arbitrary size continuous variables
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #create the logisitic regression
    lr = LogisticRegression(C=0.1, penalty='l1',tol=0.01) #tell scikit-learn what parameters to use
    clf = lr.fit(X_train_scaled,y_train) #fit the regression on the test data
    
    #make predictions based on the training data
    y_pred = clf.predict(X_test_scaled)

    #print model evaluations
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    #print coefficient values for each independent variable
    coefs = clf.coef_[0]
    for i in range(len(coefs)):
        print("{0} : {1}".format(col_names[i+2],coefs[i]))
    
    
    return (scaler, clf)


def predict(scaler,clf,list_of_data,col_names,index_name,dv_name=None):
    #if the data has a column for class name, pass it as DV and we'll get rid of it
    #otherwise exclude dv
    #this is to make it easier to dump the test set in without modifications
    
    #put the data into a dataframe and index
    df = pd.DataFrame(list_of_data,columns=col_names)
    df.set_index(index_name,inplace = True)
    
    #pull the indexs out for later
    index_vals = df.index
    
    #deal with the dv (class) column if needed
    if dv_name is not None:
        X = df.drop(dv_name,axis=1).values
    else:
        X = df.values
        
    #scale
    X_scaled = scaler.transform(X)
    
    #predict probability
    y_probs = clf.predict_proba(X_scaled)[:,1]
    
    #create and return a dictionary of id to probability
    probs = {}
    for i in range(len(index_vals)):
        probs[index_vals[i]] = y_probs[i] 
    return probs
    
    
    
def import_csv_data(filename):
    #takes a csv of data and puts it in the right format to run through create_lr_model()
    #csv must have a header row. the first col should be an id
    #the second col should be the class (0 or 1)
    #the remaining columns are independent variables
    #can be continuous variables (numbers) or discrete variables (0/1)
    #no blank variables allowed!
    list_of_data = []
    with open(filename,'r') as f:
        reader = csv.reader(f)
        first = True
        for line in reader:
            if first:
                col_names = line
                first = False
            else:
                list_of_data.append(line)
        index_name = col_names[0]
        dv_name = col_names[1]
    return (list_of_data, col_names,index_name,dv_name)
    

#processes data and creates the model when you run the code
if __name__ == "__main__":
    eng = clean_text("sample_text/english_kyoto.txt")
    sp = clean_text("sample_text/spanish_kyoto.txt")
    features,sentence_dict,col_names = make_features(eng,0)
    features,sentence_dict,col_names = make_features(sp,1,features,sentence_dict)
    scaler, clf = create_lr_model(features,col_names,"index","language")
    
    
    #ucomment the following if you want to predict on some spanish language data from a different domain
    """
    sp_test = clean_text("sample_text/spanish_soledad.txt")
    test_features,test_sentence_dict,test_col_names = make_features(sp_test,1)
    probs = predict(scaler,clf,test_features,test_col_names,"index","language")
    for p in probs:
        print("{0} : {1}".format(probs[p],test_sentence_dict[p]))
    """
    
    #uncomment to model English vs everything else
    """
    eng = clean_text("sample_text/english_1984.txt")
    fr = clean_text("sample_text/french_recherche.txt")
    sp = clean_text("sample_text/spanish_soledad.txt")
    ger = clean_text("sample_text/german_metamorph.txt")
    features,sentence_dict,col_names = make_features(eng,0)
    features,sentence_dict,col_names = make_features(sp,1,features,sentence_dict)
    features,sentence_dict,col_names = make_features(fr,1,features,sentence_dict)
    features,sentence_dict,col_names = make_features(ger,1,features,sentence_dict)
    scaler, clf = create_lr_model(features,col_names,"index","language")
    """