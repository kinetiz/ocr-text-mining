#=============================================================================
### Description ###
# The cw is to perform data mining techniques to clean, extract, explore and present 
# insight/relationship of 24 ocr books which is unstructed html data. 
#=============================================================================
### Steps
#- Scrape the text from html put in Dataframe
#- Perform feature extraction to transform document text to vectors
#- Perform multidimension scaling techniques to reduce dimensions to be ready for exploration
#- Visualise the data in low dim and perform clustering to find relationship btw each book
#- Find some insight!
#=============================================================================
import os
import pandas as pd
from bs4 import BeautifulSoup as bs
import pickle
import numpy as np
import gensim

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import ward, dendrogram

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.lines as mlines




## update book title and author
TITLES = [
"DICTIONARY GREEK AND ROMAN GEOGRAPHY",
"THE HISTORY OF TACITUS",
"THE HISTORY OF THE PELOPONNESIAN WAR",
"THE HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE",
"THE HISTORY OF ROME",
"THE WHOLE GENUINE WORKS OF FLAVIUS JOSEPHUS",
"THE HISTORY OP THE DECLINE AND FALL OF THE ROMAN EMPIRE",
"THE DESCRIPTION OF GREECE.",
"THE HISTORY OF ROME",
"HISTORY OF ROME",
"THE HISTORY OF THE PELOPONNESIAN WAR",
"HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE",
"HISTORY OF THE DECLINE AND FALL OF THE ROMAN EMPIRE",
"THE ANNALS OF TACITUS",
"ROMAN HISTORY",
"THE JEWISH WAR OR, THE HISTORY OF THE DESTRUCTION OF JERUSALEM.",
"THK ANNALS OF TACITUS",
"HISTORY OF ROME",
"THE ANTIQUITIES OF THE JEWS",
"THE FIRST AND THIRTY-THIRD BOOKS OF PLINY'S NATURAL HISTORY",
"THR HISTORY OF THI THE ROMAN EMPIRE",
"THE HISTORIES CAIUS COBNELIUS TACITUS: NOTES FOR COLLEGES",
"THE HISTORY DECLINE AND FALL ROMAN EMPIRE",
"THE LEARNED AND AUTHENTIC JEWISH HISTORIAN AND CELEBRATED WARRIOR"
]

AUTHORS = [
"WILLIAM_SMITH_LLD",
"ARTHUR_MURPHY",
"WILLIAM_SMITH",
"EDWARD_GIBBON",
"TITUS_LIVIUS",
"WILLIAM_WHISTON",
"EDWARD_GIBBON",
"PAUS_ANIAS",
"THEODOR_MOMMSEN",
"GEORGE_BAKER",
"WILLIAM_SMITH",
"THOMAS_BOWDLER",
"THOMAS_BOWDLER",
"ARTHUR_MURPHY",
"WILLIAM_GORDON",
"WILLIAM_WHISTON",
"ARTHUR_MURPHY",
"GEORGE_BAKER",
"FLAVIUS_JOSEPHU",
"JOHN_BOSTOCK",
"J_F_Dove",
"K_TYLER",
"EDWARD_GIBBON",
"FLAVIUS_JOSEPHU"
]

###### Scrap html to dataframe "books"
def scrap_html():
    walk_dir = "G:\\work\\ocr-text-mining\\gap-html\\gap-html\\"
    books = pd.DataFrame(columns=['book_name', 'contents'])
    
    print('walk_dir = ' + walk_dir)
    print('walk_dir (absolute) = ' + os.path.abspath(walk_dir))
    
    book_id = 1
    for root, subdirs, files in os.walk(walk_dir):
        path, book_name = os.path.split(root)
        all_text = ""
        for filename in files:
            # os.path.join => join str with //
            file_path = os.path.join(root, filename)
            soup = bs(open(file_path), "html.parser") #read file
            ocr_tags = soup.select(".ocr_cinfo") #read tags
            text_list = [tag.get_text() for tag in ocr_tags] #extract text
            text = ' '.join(text_list)
            if text.strip() != '':
                all_text = all_text + text + ' ' #concat text in each page and add to dataframe        
    #        print(book_name)
        
        #write all pages for processed book to dataframe
        books.loc[book_id] = [book_name, all_text]
        print("Save book: "+book_name)
        book_id+=1
    return books

        
##-- Serialize obj to file    
def save(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return 1

##-- Load obj from file    
def load(filename):
    with open(filename, 'rb') as input: 
        obj = pickle.load(input)
    return obj      
 
##--token and stem - prepare for tf-idf
def tokenize_and_stem(text):
    #initiate tokenizer and capture only English letters
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    stemmer = PorterStemmer()
    #lower and tokenize 
    tokens = tokenizer.tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems


      
##End Functions--------------------------------  
    
###====Modules=========================================
# =============================================================================
# Input: text
# Return: list of string (tokens)
# =============================================================================
def preProcess(text):
     #-- clean data
     #lower and tokenize
     tokens = tokenize_and_stem(text)
     
     #remove stop words 
     eng_stopwords = stopwords.words('english') 
     tokens_no_stop = [word for word in tokens if word not in eng_stopwords]     
     return tokens_no_stop

# =============================================================================
# Input: text list.
# Return: feature_vectors (numpy)
# =============================================================================
def tfidf(text_list,min=1,max=1.0):
    tfidf_vectorizer = TfidfVectorizer(
                            max_df=max, max_features=None,
                            min_df=min, stop_words='english',
                            use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
    tfidf_books = tfidf_vectorizer.fit_transform(text_list) #fit the vectorizer to synopses
    return tfidf_books
    #save(tfidf_vectorizer,'tfidf_vectorizer_max0.9_min0.1_ngram1-3.pkl')
    #save(tfidf_books,'tfidf_max0.9_min0.1_ngram1-3.pkl')
    
# =============================================================================
# Input: text list.
# Return: doc2vec model
# ============================================================================= 
def doc2vec(text_list, output_size=2, epoch=100):
    tagged_docs = []
    ## prepare doc2vec input - list of taggedDocument
    #for index,text in enumerate(text_list):
    #    docTokens = preProcess(text)   
    #    print("tagging:"+str(index))
    #    tagged_docs.append(gensim.models.doc2vec.TaggedDocument(docTokens, [str(index)+"_"+AUTHORS[index]]))
    #load
    tagged_docs = load('tagged_docs_author.pkl')
    #save(tagged_docs,'tagged_docs_author.pkl')
        
    # setup configurations
    d2vm = gensim.models.Doc2Vec(vector_size=output_size, min_count=0, alpha=0.025, min_alpha=0.025)
    d2vm.build_vocab(tagged_docs)
    
    print("Training doc2vec model..")
    # Train the doc2vec model
    for epoch in range(epoch):    # number of epoch
         print("training ep:"+str(epoch))
         d2vm.train(tagged_docs, total_examples=len(tagged_docs), epochs=1 )
         # Change learning rate for next epoch (start with large num to speed up at first and then decrease to fine grain learning)
         d2vm.alpha -= 0.002
         d2vm.min_alpha = d2vm.alpha
   # d2vm.train(tagged_docs, total_examples=len(tagged_docs), epochs=epoch )
    print("Done training..")
    ##d2vm.save('doc2vec.model')
    return d2vm

def doc2vec_to_vectors(d2vm): 
    # Extract vectors from doc2vec model
    feature_vectors = []
    for i in range(0,len(d2vm.docvecs)) :
        feature_vectors.append(d2vm.docvecs[i])
    
    return feature_vectors    
    

# =============================================================================
# Reduce dimensions
# Input: distance of each vectors as matrix
# Return: 2 dim vectors x[],y[]
# =============================================================================
def multidimScale(distanceVectors,mode="precomputed"):
    # multidimension scaling - metric:True=Mds, False=Nmds
    if mode != "precomputed":
        mode = "euclidean"
    mds = MDS(n_components=2, random_state=1,dissimilarity=mode, metric=True)
    pos = mds.fit_transform(distanceVectors)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    lowDimVecs = pd.DataFrame({'dim1':xs, 'dim2':ys})
    return lowDimVecs
def multidimScale3(distanceVectors,mode="precomputed"):
    # multidimension scaling - metric:True=Mds, False=Nmds
    if mode != "precomputed":
        mode = "euclidean"
    mds = MDS(n_components=3, random_state=1,dissimilarity=mode, metric=True)
    pos = mds.fit_transform(distanceVectors)  # shape (n_components, n_samples)
    xs, ys, zs = pos[:, 0], pos[:, 1], pos[:, 2]
    lowDimVecs = pd.DataFrame({'dim1':xs, 'dim2':ys, 'dim3':zs})
    return lowDimVecs

def tsne(featureVector,m = "precomputed"):
    ts = TSNE(n_components=2, random_state=1, metric=m)
    pos = ts.fit_transform(featureVector)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]
    lowDimVecs = pd.DataFrame({'dim1':xs, 'dim2':ys})
    return lowDimVecs

def svd(featureVector):
    sv = TruncatedSVD(n_components=2, random_state=1)
    pos = sv.fit_transform(bookFeatures)  
    xs, ys = pos[:, 0], pos[:, 1]
    lowDimVecs = pd.DataFrame({'dim1':xs, 'dim2':ys})
    return lowDimVecs
def pca():
    return  0

# =============================================================================
# Clustering
# Input: feature vectors
# Return: km objects
# =============================================================================
def kmean(featureVectors,k=6):    
    km = KMeans(n_clusters=k,random_state =1)
    km.fit(featureVectors)
    return km


def hc():
    linkage_matrix = ward(distVectors) #define the linkage_matrix using ward clustering pre-computed distances
    
    fig, ax = plt.subplots() # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels="title");
    
    plt.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    
    plt.tight_layout() #show plot with tight layout
    
    #uncomment below to save figure
    #plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters

            

##Start program here----------------------------------------------------------------

##======= 1) Extract data=======
#books = scrap_html()
print ("Loading books data..")
books = load('books.pkl')


##======= 2) Extract feature vectors=======
print ("Extracting book features.. ")
##------ Doc2vec---------
d2vm = doc2vec(books['contents'].tolist(),output_size=300,epoch=100)
#d2vm.save('doc2vec_e100_size300.model')#save 
bookFeatures = doc2vec_to_vectors(d2vm)

#xs = [i[0] for i in bookFeatures]
#ys = [i[1] for i in bookFeatures]
#
#lowDimVecs = pd.DataFrame({'dim1':xs, 'dim2':ys})

# ##-----TFIDF (noneed preprocess its already in tfidf function)----------
# bookFeatures = tfidf(books['contents'].tolist(),0.1,0.9)
##vectorizer = load('tfidf_vectorizer_max0.9_min0.1_ngram1-3.pkl')

bookFeatures = load('tfidf_max0.9_min0.1_ngram1-3.pkl')
distVectors = 1-cosine_similarity(bookFeatures)
mds = MDS(n_components=2000, random_state=1,dissimilarity="precomputed", metric=True)
bookFeatures = mds.fit_transform(distVectors)  # shape (n_components, n_samples)
#xs, ys = pos[:, 0], pos[:, 1]
#lowDimVecs = pd.DataFrame({'dim1':xs, 'dim2':ys})
    
##======= 3) Cluster =======

# =============================================================================
# k means determine k
# =============================================================================
X = bookFeatures
distorsions = []
for k in range(1, 24):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorsions.append(kmeans.inertia_)

fig = plt.figure()
plt.plot(range(1, 24), distorsions)
plt.scatter(range(1, 24), distorsions,color='g',marker='x')
plt.grid(True)
plt.xlabel('K')
plt.ylabel('Sum of square distance to centroids')
plt.title('Elbow curve for Doc2vec vectors')

##Process
print ("Clustering books..")
km = kmean(bookFeatures,6)
clusters = km.labels_.tolist()
bookFeatures_with_c = np.vstack([bookFeatures, km.cluster_centers_])



##======= 4) Reduce dimension=======
print ("Reducing feature dimension..")
# calculate distance between each vector
distVectors = 1-cosine_similarity(bookFeatures_with_c)
lowDimVec = multidimScale(distVectors)
centeroids = lowDimVec[24:]
lowDimVec = lowDimVec[0:24]
#assign title, authors and cluster-labels
lowDimVec = lowDimVec.assign(cluster=clusters,title=TITLES,author=AUTHORS )


###---- test many ways of dim reduction
lowDimVec = multidimScale(bookFeatures_with_c)
lowDimVec = tsne(cosine_similarity(bookFeatures))
lowDimVec = svd(bookFeatures_with_c)  
centeroids = lowDimVec[24:]
lowDimVec = lowDimVec[0:24]
lowDimVec = lowDimVec.assign(cluster=clusters,title=TITLES,author=AUTHORS )


##======= 5) Plot =======
print ("Ploting..")
color=['r','g','b','m','y','k','b','r','g']
marker=['x','o','^','v',',','p','+','<','>']
#2 dim plot
fig, ax = plt.subplots()
for i in range(0, len(lowDimVec)):
    x = lowDimVec.loc[i].dim1
    y = lowDimVec.loc[i].dim2
    c = color[int(lowDimVec.loc[i].cluster)]
    m = marker[int(lowDimVec.loc[i].cluster)]  
    a = lowDimVec.loc[i].author
    t = lowDimVec.loc[i].title
    ax.scatter(x,y,c=c,marker='o',alpha=0.8,s=100)
    ax.annotate(i+1, (x,y), fontsize=15)
ax.scatter(centeroids[:].dim1,centeroids[:].dim2,c=color,marker='x',s=100)

cir = mlines.Line2D([], [], color='k',alpha=0.8, marker='o', linestyle='None',
                          markersize=10, label='Cluster')
cross = mlines.Line2D([], [], color='k',alpha=0.8, marker='x', linestyle='None',
                          markersize=10, label='Centroid')
plt.legend(handles=[cir, cross])

ax.set_xlabel('Dimension-1')
ax.set_ylabel('Dimension-2')
plt.title('MDS mapping for TF-IDF vectors')
plt.show()

###3 dim plot
#threeDimVec = multidimScale3(distVectors)
#centeroids3 = threeDimVec[24:30]
#threeDimVec = threeDimVec[0:24]
##assign title, authors and cluster-labels
#threeDimVec = threeDimVec.assign(cluster=clusters,title=TITLES,author=AUTHORS )
#
#fig3 = plt.figure()
#ax3 = Axes3D(fig3)
##fig, ax = plt.subplots()
#for i in range(0, len(threeDimVec)):
#    x = threeDimVec.loc[i].dim1
#    y = threeDimVec.loc[i].dim2
#    z = threeDimVec.loc[i].dim3
#    c = color[int(threeDimVec.loc[i].cluster)]
#    m = marker[int(threeDimVec.loc[i].cluster)]  
#    a = threeDimVec.loc[i].author
#    t = threeDimVec.loc[i].title
#    ax3.scatter(x,y,z,c=c,marker='o',s=100)
#    #ax.text(x, y, z, a, color='red')
#    #ax3.annotate(a, (x,y,z), fontsize=7)
#ax3.set_xlabel('Dimension-1')
#ax3.set_ylabel('Dimension-2')
#ax3.set_zlabel('Dimension-3')
#plt.title('3D-plot Doc2vec K-mean Clustering with K = 6')

##======= 6) Output result =======
lowDimVec.to_csv('result_doc2vec_ep100_size300_K6.csv')


##test
# # Test the model
# #to get most similar document with similarity scores using document- name
#sims = d2vm.docvecs.most_similar("2_"+AUTHORS[1], topn=3)
#print('top similar document for gap_-C0BAAAAQAAJ: ')
#print(sims)
# 
#similar_words = d2vm.docvecs.most_similar(positive=[d2vm.docvecs['tag_2']])
#print(similar_words)
## =============================================================================





from scipy.cluster.hierarchy import ward, dendrogram
titles = "Hierarchical Clustering"
linkage_matrix = ward(distVectors) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots() # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters









from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
X = [[i] for i in [2, 8, 0, 4, 1, 9, 9, 0]]

Z = linkage(bookFeatures, 'average')
fig = plt.figure()
dn = dendrogram(Z, color_threshold = 21)
c, coph_dists = cophenet(Z, pdist(bookFeatures))
print(c)


Z = linkage(bookFeatures, 'centroid')
fig = plt.figure()
dn = dendrogram(Z, color_threshold = 0.6)
c, coph_dists = cophenet(Z, pdist(bookFeatures))
print(c)


Z = linkage(bookFeatures, 'median')
fig = plt.figure()
dn = dendrogram(Z)
c, coph_dists = cophenet(Z, pdist(bookFeatures))
print(c)


Z = linkage(bookFeatures, 'ward')
fig = plt.figure()
dn = dendrogram(Z)
c, coph_dists = cophenet(Z, pdist(bookFeatures))
print(c)




plt.show()