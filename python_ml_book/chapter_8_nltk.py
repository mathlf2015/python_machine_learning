#data clean
import pandas as pd
df = pd.read_csv('E:/mydata/movie_data.csv',encoding='ISO8859-1')
import re
def preprocessor(text):
    #去掉HTML标记
    text = re.sub('<[^>]*>', '', text)
    #找出表情符号
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower())+ ''.join(emoticons).replace('-', '')
    return text
df['review'] = df['review'].apply(preprocessor)

#将词和词根联系的分词
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

#直接分词
def tokenizer(text):
    return text.split()

#去除常用的无意义的词
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
stop = stopwords.words('english')

###################################################################################################
#25,000 documents for training and 25,000 documents for testing
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values


from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
lr_tfidf = Pipeline([('vect', tfidf),('clf',LogisticRegression(random_state=0))])

param_grid = [{'vect__ngram_range': [(1,1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer,tokenizer_porter],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]},
                {'vect__ngram_range': [(1,1)],
                'vect__stop_words': [stop, None],
                'vect__tokenizer': [tokenizer,tokenizer_porter],
                'vect__use_idf':[False],
                'vect__norm':[None],
                'clf__penalty': ['l1', 'l2'],
                'clf__C': [1.0, 10.0, 100.0]}
                ]

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,scoring='accuracy',cv=5, verbose=1,n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f'% gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f'% clf.score(X_test, y_test))