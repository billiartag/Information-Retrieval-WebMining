import json#baca json
import re#remove re di teks macem html
import html
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #sastrawi stemming
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory#sastrawi stopword
from collections import Counter

import os,shutil
import numpy as np
from nltk.tokenize import word_tokenize 
import math

# create stemmer
factoryStem = StemmerFactory()
factoryRemove = StopWordRemoverFactory()
stemmer = factoryStem.create_stemmer()
stopword = factoryRemove.create_stop_word_remover()

print("==============================================")
#keyword
alfa = 0.3
query = input("Masukkan query yang anda inginkan: ")
jumlah_hasil = input("Masukkan jumlah hasil yang anda inginkan: ")

print("==============================================")
#ambil file
list_folder = [
"Abednego Sion Hortalanus - 217116565",
"Alfonsus Yves Modiano - 217116571",
"Alvin Sucita - 217116573",
"Cosmas Yustianus Billiarta Gondokusuma - 217116585",
"Ivan Marcellino - 217116609",
"Raymond Adi P - 217116638",
"Richard Ivan - 217116647",
"Sandy Khosasi - 217116652",
"Stella Vania - 217116657",
"Stevanus Billy - 217116658",
"Vincent - 217116664",
"Wenly Poha - 217116666",
"William Adi Nata Mulianto - 217116667",
"William Hartanto - 217116668"
]

list_folder = list_folder[:5]
data =None
data_raw = None
for i in list_folder:
    with open(i+"/processed.json", encoding='utf-8') as file: 
        print(i+"/processed.json")
        # try:
        if data is None:
            data = json.load(file)
        else:
            for j in json.load(file):
                data.append(j)
    with open(i+"/raw.json", encoding='utf-8') as file: 
        print(i+"/raw.json")
        # try:
        if data_raw is None:
            data_raw = json.load(file)
        else:
            for j in json.load(file):
                data_raw.append(j)
    print(len(data))
    print(len(data_raw))

proc_content =[]
proc_title=[]
#preprocessing
##hapusin html tag
ctr=0
for i in data:
   try:
      dummy = i
      # print(ctr)
      ctr+=1
      #pp title
      stem_title =stemmer.stem(dummy['title']) 
      stop_title = stopword.remove(stem_title)
      stop_title = re.sub(r'\d+','',stop_title)
      #pp content
      teks_content = html.unescape(dummy['content'])
      stem_content =stemmer.stem(teks_content) 
      stop_content = stopword.remove(stem_title)
      stop_content = re.sub(r'\d+','',stop_content)
      proc_content.append(word_tokenize(stop_content))
      proc_title.append(word_tokenize(stop_title))
   except:
      pass
N = len(proc_title)
print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
print("Jumlah dokumen yang bisa diproses: ",N)
print("ɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅɅ")
##masukin kata kedalem list biar tau jumlah kata di tulisan ada berapa
DF = {}

for i in range(N):
    tokens = proc_content[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}

    tokens = proc_title[i]
    for w in tokens:
        try:
            DF[w].add(i)
        except:
            DF[w] = {i}
for i in DF:
    DF[i] = len(DF[i])
# print(DF)
#keywords
total_vocab = [x for x in DF] #dapetin unique keyword yang ada di teks
# print(total_vocab)
total_vocab_size = len(DF)

#DF
def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

#hitung TF IDF Content
doc = 0

tf_idf = {}

for i in range(N):
    
    tokens = proc_content[i]
    
    counter = Counter(tokens + proc_title[i])
    words_count = len(tokens + proc_title[i])
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((N+1)/(df+1))
        
        tf_idf[doc, token] = tf*idf

    doc += 1

# print(tf_idf)

#TFIDF title
doc = 0
tf_idf_title = {}

for i in range(N):
    
    tokens = proc_title[i]
    counter = Counter(tokens + proc_content[i])
    words_count = len(tokens + proc_content[i])

    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = np.log((N+1)/(df+1)) #numerator is added 1 to avoid negative values
        
        tf_idf_title[doc, token] = tf*idf

    doc += 1

# print(tf_idf_title)

# tf_idf[(0,"gas")]
# tf_idf_title[(0,"gas")]
#tf idf weighting
for i in tf_idf:
    tf_idf[i] *= alfa
for i in tf_idf_title:
    tf_idf[i] = tf_idf_title[i]

def matching_score(k, query):
    #k = jumlah dokumen
    #query = query
    stem_q =stemmer.stem(query) 
    stop_q = stopword.remove(stem_q)
    tokens = word_tokenize(stop_q)

    print("\n==============================================")
    print("Dokumen dengan Matching Score")

    query_weights = {}

    for key in tf_idf:
        
        if key[1] in tokens:
            try:
                query_weights[key[0]] += tf_idf[key]
            except:
                query_weights[key[0]] = tf_idf[key]

    query_weights = sorted(query_weights.items(), key=lambda x: x[1], reverse=True)

    print("")

    l = []

    for i in query_weights[:k]:
        l.append(i[0])

    print(l)
    return l
   

index_cocok = matching_score(int(jumlah_hasil), query)
for j in index_cocok:
    print(data[j]['title'])

#cosine similiarity

def cosine_sim(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

D = np.zeros((N, total_vocab_size))
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass

def gen_vector(tokens):
    Q = np.zeros((len(total_vocab)))
    
    counter = Counter(tokens)
    words_count = len(tokens)
    
    for token in np.unique(tokens):
        
        tf = counter[token]/words_count
        df = doc_freq(token)
        idf = math.log((N+1)/(df+1))

        try:
            ind = total_vocab.index(token)
            Q[ind] = tf*idf
        except:
            pass
    return Q

def cosine_similarity(k, query):
    stem_q =stemmer.stem(query) 
    stop_q = stopword.remove(stem_q)
    tokens = word_tokenize(stop_q)

    print("\n==============================================")
    print("Dokumen dengan Cosine Similarity")
    
    d_cosines = []
    
    query_vector = gen_vector(tokens)
    
    for d in D:
        d_cosines.append(cosine_sim(query_vector, d))
        
    out = np.array(d_cosines).argsort()[-k:][::-1]
    return out
    
index_cosine = cosine_similarity(int(jumlah_hasil), query)

jenis_topik = {}
arr_title = []
for j in index_cosine:
    print(data[j]['title'])
    arr_title.append(data[j]['title'])
    try:
        jenis_topik[data[j]['topic']]= jenis_topik[data[j]['topic']]+1
    except:
        jenis_topik[data[j]['topic']] = 1

print("\n==============================================")
print("Topik yang masuk dalam query ini:")
for i in jenis_topik:
    print (i)

#generate file HTML
path_export = "HTML/"

html_data = []
for i in index_cosine:
    html_data.append(data_raw[i])
    
#buang isi folder
for filename in os.listdir(path_export):
    file_path = os.path.join(path_export, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

#generate file
ctr=0
for i in html_data:
    fn = arr_title[ctr]
    filename=""

    for ch in fn:
        if ch.isalnum():
            filename+=ch

    file = open(path_export+filename+'.html',"w", encoding='utf-8')
    file.write(i.get('html'))
    file.close()
    ctr+=1

print("HTML file sudah di generate di folder HTML")