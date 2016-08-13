#raw term frequencies
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
            'The sun is shining',
             'The weather is sweet',
            'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())


#downweight those frequently occurring words in the feature vectors
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
#保留小数点后两位
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())




##################################################################
#将词和词根联系
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
print(tokenizer_porter('runners like running and thus they run'))

#去除常用的无意义的词
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])