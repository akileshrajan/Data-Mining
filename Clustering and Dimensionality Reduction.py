
# coding: utf-8


# ## Name: Akilesh R. Student ID.1001091662
# ## Name: Ashwin R. Student ID.1001098716 
# ## Name: Anirudh R. Student ID.1001051262
# ## Code Implementation based on Lecture from Saravanan Thirumuruganathan, University of Texas at Arlington
# 


# In[3]:

####################Do not change anything below
get_ipython().magic(u'matplotlib inline')

#Array processing
import numpy as np

#Data analysis, wrangling and common exploratory operations
import pandas as pd
from pandas import Series, DataFrame

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.display import Image
from IPython.display import display

import sklearn.datasets as datasets
from sklearn.utils import shuffle

from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import metrics

#A different implementation of k-means
import scipy as sp
import scipy.cluster.vq
import scipy.spatial.distance
from scipy.spatial.distance import cdist, pdist

from sklearn.datasets import fetch_mldata, fetch_olivetti_faces
from sklearn.utils import shuffle 
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA

import time
from collections import defaultdict

#################Might require installation of new packages
#Install selenium. Upgrade it if you have previously installed as recent version has some important bug fixes
import selenium.webdriver as webdriver
from selenium.webdriver.common.by import By

import json
from urlparse import parse_qs, urlsplit, urlunsplit

from numpy.linalg import LinAlgError
#######################End imports###################################


# In[4]:

#Set the colormap to jet. Most of the time this is good enough
# See http://matplotlib.org/examples/color/colormaps_reference.html for details
plt.set_cmap('jet')


# # Part 1. Evaluation of $k$-Means over Diverse Datasets
# 


# In[5]:

#task t1a


def part1_plot_clustering(original_data, original_cluster_assignments, 
                              kmeanspp_cluster_assignments, kmeans_random_cluster_assignments):
    plt.figure()
    fig,axes = plt.subplots(1, 3, figsize=(15,4))
    ## call scatter plot function on axes[0]
    axes[0].scatter(original_data[:, 0],original_data[:, 1], c=original_cluster_assignments)
    axes[0].set_title('Original')
    
    ## call scatter plot function on axes[1]
    axes[1].scatter(original_data[:, 0],original_data[:, 1], c=kmeanspp_cluster_assignments)
    axes[1].set_title('Init=K-Means++')
    
    ## call scatter plot function on axes[2]
    axes[2].scatter(original_data[:, 0],original_data[:, 1], c=kmeans_random_cluster_assignments)
    axes[2].set_title('Init=Random')


# In[6]:


#CCreate a dataset with 200 2-D points with 4 cluster with a standard deviation of 1.0
t1b_data, t1b_ground_truth = datasets.make_blobs(n_samples=200, centers=4, n_features=2, random_state = 1234)
#Call Kmeans with 4 clusters, with k-means++ initialization heuristic and random state of 1234
t1b_kmeanspp = KMeans(n_clusters=4, init='k-means++', random_state = 1234)
#Print the centroids
print t1b_kmeanspp.fit(t1b_data).cluster_centers_
#Find the cluster assignments for the data
t1b_kmeanspp_cluster_assignments = t1b_kmeanspp.fit(t1b_data).labels_
#Call Kmeans with 4 clusters, with random initialization heuristic and random state of 1234
t1b_kmeans_random = KMeans(n_clusters=4, init='random', random_state = 1234)
#Find the cluster assignments for the data
t1b_kmeans_random_cluster_assignments = t1b_kmeans_random.fit(t1b_data).labels_
part1_plot_clustering(t1b_data, t1b_ground_truth, t1b_kmeanspp_cluster_assignments, t1b_kmeans_random_cluster_assignments)


# In[107]:

#task t1c
# Create a dataset (make_blobs) with 200 2-D points with 4 cluster with a standard deviation of 5.0
# 


t1c_data, t1c_ground_truth = datasets.make_blobs(n_samples=200, centers=4, n_features=2, cluster_std=5.0, random_state = 1234)

t1c_kmeanspp = KMeans(n_clusters=4, init='k-means++', random_state = 1234)

t1c_kmeanspp_cluster_assignments = t1c_kmeanspp.fit(t1c_data).labels_

t1c_kmeans_random = KMeans(n_clusters=4, init='random', random_state = 1234)

t1c_kmeans_random_cluster_assignments = t1c_kmeans_random.fit(t1c_data).labels_
part1_plot_clustering(t1c_data, t1c_ground_truth, t1c_kmeanspp_cluster_assignments, t1c_kmeans_random_cluster_assignments)


# In[108]:

#task t1d
# Create a dataset (make_blobs) with 200 2-D points with 10 clusters and with a standard deviation of 1.0


t1d_data, t1d_ground_truth = datasets.make_blobs(n_samples=200, centers=10, n_features=2, random_state = 1234)
t1d_kmeanspp = KMeans(n_clusters=10, init='k-means++', random_state = 1234)
t1d_kmeanspp_cluster_assignments = t1d_kmeanspp.fit(t1d_data).labels_
t1d_kmeans_random = KMeans(n_clusters=10, init='random', random_state = 1234)
t1d_kmeans_random_cluster_assignments = t1d_kmeans_random.fit(t1d_data).labels_
part1_plot_clustering(t1d_data, t1d_ground_truth, t1d_kmeanspp_cluster_assignments, t1d_kmeans_random_cluster_assignments)




# In[109]:

#task t1e
# Create a dataset (make_blobs) with 200 2-D points with 10 clusters and with a standard deviation of 5.0

#Call K-Means with k=10

t1e_data, t1e_ground_truth = datasets.make_blobs(n_samples=200, centers=10, n_features=2, cluster_std=5.0, random_state = 1234)
t1e_kmeanspp = KMeans(n_clusters=10, init='k-means++', random_state = 1234)
t1e_kmeanspp_cluster_assignments = t1e_kmeanspp.fit(t1e_data).labels_
t1e_kmeans_random = KMeans(n_clusters=10, init='random', random_state = 1234)
t1e_kmeans_random_cluster_assignments = t1e_kmeans_random.fit(t1e_data).labels_
part1_plot_clustering(t1e_data, t1e_ground_truth, t1e_kmeanspp_cluster_assignments, t1e_kmeans_random_cluster_assignments)




# In[110]:

#task t1f

#Ref: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html 
# Create a dataset with 200 2-D points 

# call the K-Means function with k=2

t1f_data, t1f_ground_truth = datasets.make_circles(n_samples=200, random_state = 1234)
t1f_kmeanspp = KMeans(n_clusters=2, init='k-means++', random_state = 1234)
t1f_kmeanspp_cluster_assignments = t1f_kmeanspp.fit(t1f_data).labels_
t1f_kmeans_random = KMeans(n_clusters=2, init='random', random_state = 1234)
t1f_kmeans_random_cluster_assignments = t1f_kmeans_random.fit(t1f_data).labels_

part1_plot_clustering(t1f_data, t1f_ground_truth, t1f_kmeanspp_cluster_assignments, t1f_kmeans_random_cluster_assignments)




#task t1g
#Ref: http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html 
# Create a dataset with 200 2-D points 
#call the K-Means function with k=2

t1g_data, t1g_ground_truth = datasets.make_moons(n_samples=200, random_state=1234)
t1g_kmeanspp = KMeans(n_clusters=2, init='k-means++', random_state = 1234)
t1g_kmeanspp_cluster_assignments = t1g_kmeanspp.fit(t1g_data).labels_
t1g_kmeans_random = KMeans(n_clusters=2, init='random', random_state = 1234)
t1g_kmeans_random_cluster_assignments = t1g_kmeans_random.fit(t1g_data).labels_
part1_plot_clustering(t1g_data, t1g_ground_truth, t1g_kmeanspp_cluster_assignments, t1g_kmeans_random_cluster_assignments)


# ###$k$-Means and Image Compression via Vector Quantization
# 


# In[112]:


#Code courtesy: Sklearn
#china is a 3-D array where the first two dimensions are co-ordinates of pixels (row and column)
# and third coordinate is a tuple with 3 values for RGB value
china = datasets.load_sample_image("china.jpg")
china = np.array(china, dtype=np.float64) / 255
china_w, china_h, china_d = tuple(china.shape)
print "Width=%s, Height=%s, Depth=%s" % (china_w, china_h, china_d)
#Convert it to a 2D array for analysis
china_image_array = np.reshape(china, (china_w * china_h, china_d))
print "In 2-D the shape is ", china_image_array.shape

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


plt.figure()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)

#**End Reference


#t1h:


#K-means with different values of k
# Then using the new centroids, compress the image and display it. 

t1h_start_time = time.time()
plt.figure()
fig,axes = plt.subplots(2, 2, figsize=(15,6))

#the 2d is for convenience
t1h_k_values = [[16, 32], [64,128]]
for i in range(2):
    for j in range(2):
        print "Handling k =", t1h_k_values[i][j]
        #call Kmeans with k=t1h_k_values [i][j] and random state = 1234
        t1h_kmeans_obj = KMeans(n_clusters=t1h_k_values [i][j],  random_state = 1234)
        #fit the object with china image array variable
        t1h_kmeans_fit = t1h_kmeans_obj.fit(china_image_array)
        axes[i][j].imshow(recreate_image(t1h_kmeans_fit.cluster_centers_, t1h_kmeans_fit.labels_, china_w, china_h))
        axes[i][j].set_title('Compression with' + str(t1h_k_values[i][j]) + " colors")
        
        axes[i][j].grid(False)
        axes[i][j].get_xaxis().set_ticks([])
        axes[i][j].get_yaxis().set_ticks([])

print "Clustering over entire data took %s seconds" % (time.time() - t1h_start_time)




#t1i:



t1i_china_sample = shuffle(china_image_array, random_state=1234)[:1000]


#K-means with different values of k
# Using the new centroids, compress the image and display it. 

t1i_start_time = time.time()
plt.figure()
fig,axes = plt.subplots(2, 2, figsize=(15,6))

#the 2d is for convenience
t1i_k_values = [[16, 32], [64,128]]    

for i in range(2):
    for j in range(2):
        print "Handling k =", t1i_k_values[i][j]
            
        #call Kmeans with k=t1h_k_values [i][j] and random state = 1234 
        t1i_kmeans_obj = KMeans(n_clusters=t1i_k_values[i][j],  random_state = 1234)
        #fit the object with the SAMPLE
        t1i_kmeans_fit = t1h_kmeans_obj.fit(t1i_china_sample)
        
        t1i_cluster_assignments = t1i_kmeans_fit.predict(china_image_array)
        
        axes[i][j].imshow(recreate_image(t1i_kmeans_fit.cluster_centers_, t1i_cluster_assignments, china_w, china_h))
        axes[i][j].set_title('Compression with' + str(t1h_k_values[i][j]) + " colors")

        axes[i][j].grid(False)
        axes[i][j].get_xaxis().set_ticks([])
        axes[i][j].get_yaxis().set_ticks([])
        

print "Clustering with Sampling took %s seconds" % (time.time() - t1i_start_time)




#Do not change anything below

#All three functions were adapted from the code by Reid Johnson from University of Notre Dame

def compute_ssq(data, k, kmeans):
    dist = np.min(cdist(data, kmeans.cluster_centers_, 'euclidean'), axis=1)
    tot_withinss = sum(dist**2) # Total within-cluster sum of squares
    totss = sum(pdist(data)**2) / data.shape[0] # The total sum of squares
    betweenss = totss - tot_withinss # The between-cluster sum of squares
    return betweenss/totss*100
    



def ssq_statistics(data, ks, ssq_norm=True):
    ssqs = sp.zeros((len(ks),)) # array for SSQs (lenth ks)
    
    for (i,k) in enumerate(ks): # iterate over the range of k values
        kmeans = KMeans(n_clusters=k, random_state=1234).fit(data)
        
        if ssq_norm:
            ssqs[i] = compute_ssq(data, k, kmeans)
        else:
            # The sum of squared error (SSQ) for k
            ssqs[i] = kmeans.inertia_
    return ssqs


def gap_statistics(data, refs=None, nrefs=20, ks=range(1,11)):
    
    sp.random.seed(1234)
    shape = data.shape
    dst = sp.spatial.distance.euclidean
    
    if refs is None:
        tops = data.max(axis=0) # maxima along the first axis (rows)
        bots = data.min(axis=0) # minima along the first axis (rows)
        dists = sp.matrix(sp.diag(tops-bots)) # the bounding box of the input dataset
        
        # Generate nrefs uniform distributions each in the half-open interval [0.0, 1.0)
        rands = sp.random.random_sample(size=(shape[0],shape[1], nrefs))
        
        # Adjust each of the uniform distributions to the bounding box of the input dataset
        for i in range(nrefs):
            rands[:,:,i] = rands[:,:,i]*dists+bots
    else:
        rands = refs
        
    gaps = sp.zeros((len(ks),))   # array for gap statistics (lenth ks)
    errs = sp.zeros((len(ks),))   # array for model standard errors (length ks)
    difs = sp.zeros((len(ks)-1,)) # array for differences between gaps (length ks-1)

    for (i,k) in enumerate(ks): # iterate over the range of k values
        # Cluster the input dataset via k-means clustering using the current value of k
        try:
            (kmc,kml) = sp.cluster.vq.kmeans2(data, k)
        except LinAlgError:
            #kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(data)
            kmeans = KMeans(n_clusters=k).fit(data)
            (kmc, kml) = kmeans.cluster_centers_, kmeans.labels_

        # Generate within-dispersion measure for the clustering of the input dataset
        disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])

        # Generate within-dispersion measures for the clusterings of the reference datasets
        refdisps = sp.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            # Cluster the reference dataset via k-means clustering using the current value of k
            try:
                (kmc,kml) = sp.cluster.vq.kmeans2(rands[:,:,j], k)
            except LinAlgError:
                #kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(rands[:,:,j])
                kmeans = KMeans(n_clusters=k).fit(rands[:,:,j])
                (kmc, kml) = kmeans.cluster_centers_, kmeans.labels_

            refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])

        # Compute the (estimated) gap statistic for k
        gaps[i] = sp.mean(sp.log(refdisps) - sp.log(disp))

        # Compute the expected error for k
        errs[i] = sp.sqrt(sum(((sp.log(refdisp)-sp.mean(sp.log(refdisps)))**2)                             for refdisp in refdisps)/float(nrefs)) * sp.sqrt(1+1/nrefs)

    # Compute the difference between gap_k and the sum of gap_k+1 minus err_k+1
    difs = sp.array([gaps[k] - (gaps[k+1]-errs[k+1]) for k in range(len(gaps)-1)])

    #print "Gaps: " + str(gaps)
    #print "Errs: " + str(errs)
    #print "Difs: " + str(difs)

    return gaps, errs, difs




#t1j


#Interpreting the charts:
#  Elbow method: http://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#The_Elbow_Method
#  Gap Statistics: $k$ where the first drop in trend happens
#  Gap Statistics differences: $k$ where you get the first positive values
def t1j_plot_clustering_statistics(data, k_min, k_max):
 
    plt.figure()
    fig,axes = plt.subplots(1, 4, figsize=(16, 4))
    
    #range(a,b) returns a .. b-1
    ks = range(k_min, k_max+1)
    
    #plot the data distribution as a scatter plot on axes[0] variable
    # For now ignore the color field. We will use data where #clusters is easy to see
    axes[0].scatter(data[:,0],data[:,1])
    axes[0].set_title("Original Data")
    
    ssqs = ssq_statistics(data, ks=ks)
    #create a line chart with x axis as different k values 
    #  and y-axis as ssqs on axes[1] variable
    axes[1].plot(ks, ssqs)
    axes[1].set_title("Elbow Method and SSQ")
    axes[1].set_xlabel("$k$")
    axes[1].set_ylabel("SSQ")
  
    
    
    #Do not change anything below for the rest of the function
    # Code courtesy: Reid Johnson from U. of Notre Dame
    gaps, errs, difs = gap_statistics(data, nrefs=25, ks=ks)
    
    max_gap = None
    if len(np.where(difs > 0)[0]) > 0:
        max_gap = np.where(difs > 0)[0][0] + 1 # the k with the first positive dif
    if max_gap:
        print "By gap statistics, optimal k seems to be ", max_gap
    else:
        print "Please use some other metrics for finding k"
        
     #Create an errorbar plot
    rects = axes[2].errorbar(ks, gaps, yerr=errs, xerr=None, linewidth=1.0)

    #Add figure labels and ticks
    axes[2].set_title('Clustering Gap Statistics')
    axes[2].set_xlabel('Number of clusters k')
    axes[2].set_ylabel('Gap Statistic')
    axes[2].set_xticks(ks)
    # Add figure bounds
    axes[2].set_ylim(0, max(gaps+errs)*1.1)
    axes[2].set_xlim(0, len(gaps)+1.0)

    ind = range(1,len(difs)+1) # the x values for the difs
    
    max_gap = None
    if len(np.where(difs > 0)[0]) > 0:
        max_gap = np.where(difs > 0)[0][0] + 1 # the k with the first positive dif

    #Create a bar plot
    axes[3].bar(ind, difs, alpha=0.5, color='g', align='center')

    # Add figure labels and ticks
    if max_gap:
        axes[3].set_title('Clustering Gap Differences\n(k=%d Estimated as Optimal)' % (max_gap))
    else:
        axes[3].set_title('Clustering Gap Differences\n')
    axes[3].set_xlabel('Number of clusters k')
    axes[3].set_ylabel('Gap Difference')
    axes[3].xaxis.set_ticks(range(1,len(difs)+1))

    #Add figure bounds
    axes[3].set_ylim(min(difs)*1.2, max(difs)*1.2)
    axes[3].set_xlim(0, len(difs)+1.0)


# In[20]:


t1j_data, t1j_ground_truth = datasets.make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, random_state=1234)
t1j_plot_clustering_statistics(t1j_data, 1, 10)

t1j_data, t1j_ground_truth = datasets.make_blobs(n_samples=200, n_features=2, centers=5, cluster_std=0.5, random_state=1234)
t1j_plot_clustering_statistics(t1j_data, 1, 10)


# # Part 2. Evaluation of Hierarchical Clustering over Diverse Datasets
# 

#task t2a


def part2_plot_clustering(original_data, original_cluster_assignments, 
                              ward_linkage_cluster_assignments, complete_linkage_cluster_assignments, 
                              average_linkage_cluster_assignments):
    plt.figure()
    fig,axes = plt.subplots(1, 4, figsize=(16,4))
    
    ## call scatter plot function on axes[0]
    axes[0].scatter(original_data[:, 0],original_data[:, 1], c=original_cluster_assignments)
    axes[0].set_title('Original')
    
    ## call scatter plot function on axes[1]
    axes[1].scatter(original_data[:, 0],original_data[:, 1], c=ward_linkage_cluster_assignments)
    axes[1].set_title('Ward Linkage')
    
    ## call scatter plot function on axes[2]
    axes[2].scatter(original_data[:, 0],original_data[:, 1], c=complete_linkage_cluster_assignments)
    axes[2].set_title('Complete Linkage')
    
    ## call scatter plot function on axes[3]
    axes[3].scatter(original_data[:, 0],original_data[:, 1], c=average_linkage_cluster_assignments)
    axes[3].set_title('Average Linkage')    


# In[241]:

#Task t2b

## to Create a dataset with make_blobs 200 2-D points with 4 cluster with a standard deviation of 1.0
t2b_data, t2b_ground_truth = datasets.make_blobs(n_samples=200, centers=4, n_features=2, random_state = 1234)
## Call AgglomerativeClustering with 4 clusters with ward linkage
t2b_agg_ward = AgglomerativeClustering(n_clusters = 4, linkage = 'ward')
## Find the cluster assignments for the data
t2b_ward_linkage_cluster_assignments = t2b_agg_ward.fit(t2b_data).labels_


## Call AgglomerativeClustering with 4 clusters with complete linkage
t2b_agg_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete')
## Find the cluster assignments for the data
t2b_complete_linkage_cluster_assignments = t2b_agg_complete.fit(t2b_data).labels_

## Call AgglomerativeClustering with 4 clusters with average linkage
t2b_agg_average = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
## Find the cluster assignments for the data
t2b_average_linkage_cluster_assignments = t2b_agg_average.fit(t2b_data).labels_


part2_plot_clustering(t2b_data, t2b_ground_truth, t2b_ward_linkage_cluster_assignments, 
                            t2b_complete_linkage_cluster_assignments, t2b_average_linkage_cluster_assignments)


# In[243]:

#Task t2c

## to Create a dataset with make_circles function with 200 2-D points 
t2c_data, t2c_ground_truth = datasets.make_circles(n_samples=200, random_state = 1234)

## Call AgglomerativeClustering with 2 clusters with ward linkage
t2c_agg_ward = AgglomerativeClustering(n_clusters = 2, linkage = 'ward')
## Find the cluster assignments for the data
t2c_ward_linkage_cluster_assignments = t2c_agg_ward.fit(t2c_data).labels_

## Call AgglomerativeClustering with 2 clusters with complete linkage
t2c_agg_complete = AgglomerativeClustering(n_clusters = 2, linkage = 'complete')
## Find the cluster assignments for the data
t2c_complete_linkage_cluster_assignments = t2c_agg_complete.fit(t2c_data).labels_

## Call AgglomerativeClustering with 2 clusters with average linkage
t2c_agg_average = AgglomerativeClustering(n_clusters = 2, linkage = 'average')
## Find the cluster assignments for the data
t2c_average_linkage_cluster_assignments = t2c_agg_average.fit(t2c_data).labels_


part2_plot_clustering(t2c_data, t2c_ground_truth, t2c_ward_linkage_cluster_assignments, 
                            t2c_complete_linkage_cluster_assignments, t2c_average_linkage_cluster_assignments)


# In[245]:

#Task t2d

## to Create a dataset with make_moons function with 200 2-D points 
t2d_data, t2d_ground_truth = datasets.make_moons(n_samples = 200, random_state = 1234) 

## Call AgglomerativeClustering with 2 clusters with ward linkage
t2d_agg_ward = AgglomerativeClustering(n_clusters = 2, linkage = 'ward')
## Find the cluster assignments for the data
t2d_ward_linkage_cluster_assignments = t2d_agg_ward.fit(t2d_data).labels_

## Call AgglomerativeClustering with 2 clusters with complete linkage
t2d_agg_complete = AgglomerativeClustering(n_clusters = 2, linkage = 'complete')
## Find the cluster assignments for the data
t2d_complete_linkage_cluster_assignments = t2d_agg_complete.fit(t2d_data).labels_

## Call AgglomerativeClustering with 2 clusters with average linkage
t2d_agg_average = AgglomerativeClustering(n_clusters = 2, linkage = 'average')
## Find the cluster assignments for the data
t2d_average_linkage_cluster_assignments = t2d_agg_average.fit(t2d_data).labels_


part2_plot_clustering(t2d_data, t2d_ground_truth, t2d_ward_linkage_cluster_assignments, 
                            t2d_complete_linkage_cluster_assignments, t2d_average_linkage_cluster_assignments)


# In[252]:

#t2e: Let us now create and visualize dendrogram for a toy datasset

t2e_data, t2e_ground_truth = datasets.make_blobs(n_samples=20, n_features=2, centers=2, cluster_std=0.5, random_state=1234)
plt.figure()
plt.scatter(t2e_data[:, 0], t2e_data[:, 1], c=t2e_ground_truth)
plt.show()

#Plot the dendrogram of t2edata
## compute the pairwise distance 
t2e_data_dist = metrics.pairwise.pairwise_distances(t2e_data)

## compute the linkage 
t2e_data_linkage = linkage(t2e_data)

## plot the dendrogram 
t2e_data_dendrogram = dendrogram(t2e_data_linkage)


# # Part 3. Comparison of Clustering Evaluation Metrics
# 
# In the class, we mostly focused on SSE measure for evaluating how good a cluster is. There are many other statistical measures, and you will test them in this task. Broadly, they can split into two categories.
# 
# 1. Metrics for Clustering evaluation when ground truth is known
# 2. Metrics for when ground truth is not known
# 
# The following url has helpful information: http://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

# ##Ground Truth Cluster Assigments Available
# 
# Let us first consider the case where the ground truth cluster assignments are available. This is an ideal case and often in real world it is not the case (as clustering is part of unsupervised learning after all). However, since we created our datasets synthetically, it is possible to know the ground truth assignments.
# 
# In this task, you will evaluating the following metrics:
# 
# 1. Adjusted Rand index
# 2. Adjusted Mutual Information Score
# 3. Homogeneity
# 4. Completeness
# 5. V-measure score

# In[228]:

#Do not change anything below

#Let us create the data that we will be using to evaluate measures in the next cell
t3_data, t3_ground_truth = datasets.make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, random_state=1234)

t3_k_min = 1
t3_k_max = 10
t3_ind = range(t3_k_min, t3_k_max+1)


plt.figure()
plt.scatter(t3_data[:, 0], t3_data[:, 1], c=t3_ground_truth)
plt.show()


# In[229]:

#Task t3a


t3a_adjusted_rand_index_scores = [0 for _ in t3_ind]
t3a_adjusted_mutual_info_scores = [0 for _ in t3_ind]
t3a_adjusted_homogeneity_scores = [0 for _ in t3_ind]
t3a_adjusted_completeness_scores = [0 for _ in t3_ind]
t3a_adjusted_v_measure_scores = [0 for _ in t3_ind]

for k in t3_ind:
    
    ## Call KMeans with k clusters with k-means++ initialization and random state of 1234 
    t3a_kmeanspp = KMeans(n_clusters=k, init='k-means++', random_state = 1234)
    ## Find the cluster assignments for the data
    t3a_kmeanspp_cluster_assignments = t3a_kmeanspp.fit(t3_data).labels_
    
    #Now let us compute the clustering score for each metric (use metrics.xyz for getting function xyz)
    # Watch out for the argument order (true, predicted)
    
    ## compute the score based on ADJUSTED random index
    t3a_adjusted_rand_index_scores[k-1] = metrics.adjusted_rand_score(t3_ground_truth,t3a_kmeanspp_cluster_assignments)
    
    ## compute the score based on ADJUSTED mutual information score
    t3a_adjusted_mutual_info_scores[k-1] = metrics.adjusted_mutual_info_score(t3_ground_truth,t3a_kmeanspp_cluster_assignments)
    
    ## compute the score based on homogeneity score
    t3a_adjusted_homogeneity_scores[k-1] = metrics.homogeneity_score(t3_ground_truth,t3a_kmeanspp_cluster_assignments)
    
    ## compute the score based on completeness index
    t3a_adjusted_completeness_scores[k-1] = metrics.completeness_score(t3_ground_truth,t3a_kmeanspp_cluster_assignments)
    
    ## compute the score based on v-measure index
    t3a_adjusted_v_measure_scores[k-1] = metrics.v_measure_score(t3_ground_truth,t3a_kmeanspp_cluster_assignments)
    
    

plt.figure()
plt.plot(t3_ind, t3a_adjusted_rand_index_scores, label="Adjusted Rand Index")
plt.plot(t3_ind, t3a_adjusted_mutual_info_scores, label="Adjusted Mutual Info")
plt.plot(t3_ind, t3a_adjusted_homogeneity_scores, label="Homegeneity")
plt.plot(t3_ind, t3a_adjusted_completeness_scores, label="Completeness Score")
plt.plot(t3_ind, t3a_adjusted_v_measure_scores, label="V-Measure")

plt.title("$k$ vs Metrics")
plt.xlabel("$k$")
plt.ylabel("Clustering Evaluation Metrics")
plt.ylim([0.0, 1.0])
plt.legend(loc="lower right")
plt.show()


# ##Ground Truth Cluster Assigments NOT Available
# 
# Let us now consider the case where the ground truth cluster assignments is not available. Often in real world you do not know the right "answer". Let us use the synthetic data we created above (but ignore the ground truth). We will consider three simple measures that analyze how good a particular clustering is. In this task, you will evaluating the following metrics:
# 
# 1. SSQ
# 2. Silhoutte Coefficient
# 3. Stability
# 

# In[230]:

#Do not change anything below

#Code Courtesy: Derek Greene from University College, Dublin

#The following function computes pairwise stability of a list of clusterings
# the mean similarity between the clusterings as defined by a particular similarity metric. 
# In this case we use the Adjusted Rand Index to calculate the similarities.
def calc_pairwise_stability( clusterings, metric ):
    sim_values = []
    for i in range(len(clusterings)):
        for j in range(i+1,len(clusterings)):
            sim_values.append( metric( clusterings[i], clusterings[j] ) )
    return np.array( sim_values ).mean()

#Given data, take a sample, run k-means on it, make predictions
def t3_kmeans_sample( X, k, sampling_ratio ):
    # create a matrix with subset of samples
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle( indices )
    n_subset = int(n_samples * sampling_ratio) 
    X_subset = X[indices[0:n_subset]] 
    # cluster the subset
    clusterer = KMeans(n_clusters=k, n_init=1, init='random', max_iter = 100)
    clusterer.fit(X_subset)
    # produce an assignment for all samples
    return clusterer.predict(X)


# In[231]:

#Let us again use t3_data and t3_ground_truth, except we do not consider the ground truth. 

#Task t3b

t3b_ssq_scores = [0 for _ in t3_ind]
t3b_silhoutte_coefficient_scores = [0 for _ in t3_ind]
t3b_stability_scores = [0 for _ in t3_ind]

for k in t3_ind:
    
    ## Call KMeans with k clusters with k-means++ initialization and random state of 1234 
    t3b_kmeanspp = KMeans(n_clusters=k, init='k-means++', random_state = 1234)
    ## fit t3b_kmeanspp to data
    t3b_kmeanspp_fitted = t3b_kmeanspp.fit(t3_data)
    ## Find the cluster assignments for the data
    t3b_kmeanspp_cluster_assignments = t3b_kmeanspp.fit(t3_data).labels_
    
    #Now let us compute the clustering score for each metric (use metrics.xyz for getting function xyz)
    # Watch out for the argument order (true, predicted)
    
    ## compute ssq score using compute_ssq function
    t3b_ssq_scores[k-1] = compute_ssq(t3_data,t3b_kmeanspp_fitted.cluster_centers_,t3b_kmeanspp)
    
    ## compute the score based on silhouette_score with a sample size of 50
    #Note: do not set random state here - else it will give a constant score
    # Your results might look quite different from mine
    if k == 1: #Stability is defined for k >= 2
        continue
    t3b_silhoutte_coefficient_scores[k-1] = metrics.silhouette_score(t3_data,t3b_kmeanspp_cluster_assignments, sample_size=50)
    
    #Do not change: compute the score based on stability score
   
    
    #Run k-means on a small sample , make predictions based on the sample centroids and see how stable they are
    np.random.seed(1234)
    t3b_stability_clusterings = [t3_kmeans_sample( t3_data, k, 0.5 ) for run in range(10)]
    t3b_stability_scores[k-1] = calc_pairwise_stability(t3b_stability_clusterings, metrics.adjusted_rand_score)
        
    

#Do not change anything below
plt.figure()
fig,axes = plt.subplots(1, 3, figsize=(15,4))

axes[0].plot(t3_ind, t3b_ssq_scores)
axes[0].set_title('SSQ Scores')
axes[0].set_xlabel('$k$')
axes[0].set_ylabel('SSQ Scores')

axes[1].plot(t3_ind, t3b_silhoutte_coefficient_scores)
axes[1].set_title('Silhoutte Coefficient')
axes[1].set_xlabel('$k$')
axes[1].set_ylabel('Silhoutte Coefficient')
axes[1].set_ylim( (0.0, 1.0) )

axes[2].plot(t3_ind, t3b_stability_scores)
axes[2].set_title('Stability of Clusters')
axes[2].set_xlabel('$k$')
axes[2].set_ylabel('Stability')

plt.show()


# #Part 4: Clustering your Facebook Friends


#DO not change anything below

#Change to webdriver.Chrome() if Chrome is your primary browser.

driver = webdriver.Firefox()
driver.maximize_window()


SLEEP_TIME = 5



def stripParamsFromUrl(url):
    scheme, netloc, path, query_string, fragment = urlsplit(url)
    return urlunsplit((scheme, netloc, path, '', ''))

def get_likes_url(url):
    if url[-1] != "/":
        url = url + "/"
    if url.find("profile.php") >= 0:
        url = url + "&sk=likes"
    else:
        url = url + "likes"
    return url


# In[5]:

#Task t4a
def loginToFacebook(driver, user_name, password):
    ## Go to facebook.com
    driver.get("https://www.facebook.com/")
    time.sleep(SLEEP_TIME)
    ## Enter the value in user_name variable in the Email text box
    emailTextBox = driver.find_element_by_id("email")
    emailTextBox.send_keys(user_name) 
    ## Enter the value in password variable in the Password text box
    passwordTextBox = driver.find_element_by_id("pass")
    passwordTextBox.send_keys(password)
    
    passwordTextBox.submit()


loginToFacebook(driver, "user_name", "password")


# In[6]:

#DO not change anything below
def goto_profile_page(driver):
    
    elem = driver.find_element_by_css_selector("a._2dpe._1ayn")
    elem.click()
goto_profile_page(driver)


# In[7]:

#DO not change anything below
def goto_friends_page(driver):
   
    elem = driver.find_element(By.CSS_SELECTOR, "[data-tab-key='friends']")
    elem.click()    
    time.sleep(SLEEP_TIME)

#Helper code to get all your friend names and their profile url
def get_all_friend_details(driver):
    
    try:
        #Get the friends pagelet. FB pages are organized by pagelets
        # Running your find element code within a pagelet is a good idea
        pagelet = driver.find_element_by_css_selector("#pagelet_timeline_medley_friends > div[id^='collection_wrapper']")
        #Lot of you have hundreds of friends while FB only shows a small subset 
        # When you scroll down, it loads the remaining friends dynamically
        # Find how many friends are their initially
        len1 = len(pagelet.find_elements_by_css_selector("div.fsl.fwb.fcb > a"))
    except Exception as ex:
        print "Caught exception in getting friends. Try again"
        return []
    
    while True:
        try:
            #Scroll down
            driver.execute_script("window.scrollBy(0,10000)", "")
            #wait for friend details to load
            time.sleep(SLEEP_TIME)
            #Find the friends pagelet again
            #Both the browser, FB and selenium do aggressive caching
            # Sometimes, this might cause invalid references
            # Hence, getting the pagelet object fresh is a good idea
            pagelet = driver.find_element_by_css_selector("#pagelet_timeline_medley_friends > div[id^='collection_wrapper']")
            #Find how many friends you have after scrolling
            len2  = len(pagelet.find_elements_by_css_selector("div.fsl.fwb.fcb > a"))
            #If it remained the same, we have loaded all of them
            # Else repeat the process
            if len1 == len2:
                break
            len1 = len2
        except Exception as ex:
            break
    
    #Now we have a page that has all the friends
    friends = []
    try:
        #Get the pagelet object 
        pagelet = driver.find_element_by_css_selector("#pagelet_timeline_medley_friends > div[id^='collection_wrapper']")
        #Get the DOM object containing required details of your friends
        all_friends = pagelet.find_elements_by_css_selector("div.fsl.fwb.fcb > a")
        if len(all_friends) == 0:
            return []
        else:
            for i in range(len(all_friends)):
            #for i in range(60):
                #Get their name
                name = all_friends[i].get_attribute("text") 
                #Get their profile url
                url = stripParamsFromUrl(all_friends[i].get_attribute("href"))
                friends.append( {"Name": name, "ProfileURL": url})
                if i % 100 == 0:
                    print "Handled %s friends" % (i,)
    except Exception as ex:
        pass
    return friends

#Store the list of friends to a file 
def log_friend_details_to_file(friends_details, file_name):
    with open(file_name, "w") as output_file:
        #Notice how we use json library to convert the array to a string and write to a file
        json.dump(friends_details, output_file)
        


# In[8]:

#Do not change anything below
#Go to your friends page, collect their details and write it to an output file
goto_friends_page(driver)
friends_details = get_all_friend_details(driver)
log_friend_details_to_file(friends_details, "fb_friend_dtls.txt")


# In[9]:

#Task t4b: Collect the list of things your friend likes
def collect_friend_likes(driver, friend_name, friend_profile_url):
    #Directly go to likes tab of the url
    likes_url = get_likes_url(friend_profile_url)
    driver.get(likes_url)
    time.sleep(SLEEP_TIME)
    
    try:
        ## get the likes pagelet pagelet_timeline_app_collection_100001137020875:2409997254:96
        pagelet = driver.find_element_by_css_selector("#pagelet_timeline_medley_likes > div[id^='collection_wrapper']")
        ## Get the list of items liked currently
        len1 = len(pagelet.find_elements_by_css_selector("div.fsl.fwb.fcb > a"))
    except Exception as ex:
        #This person has no likes page or has not given us the permission
        return []
    
    while True:
        try:
            driver.execute_script("window.scrollBy(0,10000)", "")
            time.sleep(SLEEP_TIME)
            
            ## get the likes pagelet
            pagelet = driver.find_element_by_css_selector("#pagelet_timeline_medley_likes > div[id^='collection_wrapper']")
            ## Get the list of items liked currently
            len2  = len(pagelet.find_elements_by_css_selector("div.fsl.fwb.fcb > a"))
            
            if len1 == len2:
                break
            len1 = len2
        except Exception as ex:
            break
    
    friend_likes = []
    try:
        ## get the likes pagelet
        pagelet = driver.find_element_by_css_selector("#pagelet_timeline_medley_likes > div[id^='collection_wrapper']")
        ## Get the list of items liked currently - i.e. get the DOM object with their names
        all_friend_likes = pagelet.find_elements_by_css_selector("div.fsl.fwb.fcb > a")
        ## Get the list of items liked currently - i.e. get the DOM object with their type
        all_friend_like_types = pagelet.find_elements_by_css_selector("div._5k4f")
        
        pass
        if len(all_friend_likes) == 0:
            return []
        else:
            for i in range(len(all_friend_likes)):
                ## get the name of the item your friend liked. Eg, Bill Gates
                like_name = all_friend_likes[i].get_attribute("text") 
                ## get the type of the item your friend liked. Eg, Public Figure
                like_type = all_friend_like_types[i].text
                
                friend_likes.append( {"Item": like_name, "Type": like_type})
                
    except Exception as ex:
        pass
    
    return friend_likes



p4_friend_profile_dtls = json.loads(open("fb_friend_dtls.txt").read())


p4_offset = 0


output_file = open("fb_friend_like_dtls.txt", "a")
for i in range(p4_offset, len(p4_friend_profile_dtls)):
    friend_dtls = p4_friend_profile_dtls[i]
    friend_name, friend_profile_url = friend_dtls["Name"], friend_dtls["ProfileURL"]
    print "Handling friend %s : %s" % (i, friend_name)
    friend_like_dtls = collect_friend_likes(driver, friend_name, friend_profile_url)
    #Append friend_name so that it is findable later
    friend_like_dtls = {"Name": friend_name, "Likes":friend_like_dtls}
    json.dump(friend_like_dtls, output_file)
    output_file.write("\n")
    output_file.flush()
output_file.close()



p4_fb_friend_like_dtls = [json.loads(line) for line in open("fb_friend_like_dtls.txt").readlines()]


#Task t4c:



#We will use a nifty Python package called defaultdict.# See https://docs.python.org/2/library/collections.html#defaultdict-examples for some examples

t4c_categorized_friend_likes = defaultdict(set)

for i in range(len(p4_fb_friend_like_dtls)):
    
    #p4_friend_i_likes should now be an array of dictionaries each with two keys: 
    #  "Item": name of the item, "Type": the type of item
    p4_friend_i_likes = p4_fb_friend_like_dtls[i]["Likes"]
    
    for j in range(len(p4_friend_i_likes)):
        p4_friend_i_likes_j_th_entry = p4_friend_i_likes[j]
        
        ## assign it to name of the item
        t4c_friend_like_item_name = p4_friend_i_likes_j_th_entry["Item"]
        ## assign it to type of the item
        t4c_friend_like_item_type = p4_friend_i_likes_j_th_entry["Type"]

        ## put each item into appropriate set    
     
        t4c_categorized_friend_likes[t4c_friend_like_item_type].add(t4c_friend_like_item_name)



t4_item_categories = sorted(t4c_categorized_friend_likes.keys())
print t4_item_categories

t4_num_liked_item_categories = len(t4_item_categories)
t4_num_liked_items = 0


t4_categorized_friend_likes = defaultdict(set)
for category in t4_item_categories:
    t4_categorized_friend_likes[category] = sorted(t4c_categorized_friend_likes[category])
    t4_num_liked_items = t4_num_liked_items + len(t4_categorized_friend_likes[category])

t4_item_category_to_index_dict = {}
temp_index = 0
for category in t4_item_categories:
    for item in t4_categorized_friend_likes[category]:
        t4_item_category_to_index_dict[(category, item)] = temp_index
        temp_index += 1



# In[8]:

#Task t4d


#The three arguments are:
#   friend_like_dtls: details of a friend including his/her name and their likes
#   item_categories: sorted list of categories
#   categorized_friend_likes: a dictionary with item_categories as keys and for key has a sorted list of items
#                                  that he/she liked
# Output: a vector representation of your friends likes

def t4d_convert_friend_likes_to_vector(friend_like_dtls):
    #Initialize vector with all zeros
    friend_vector_repr = np.zeros(t4_num_liked_items)
    
    ## finish the code!
    #vector_index = 
    friend_likes = friend_like_dtls["Likes"]
    for items in friend_likes:
        #Name of the item
        like_item_name = items["Item"]
        #type of the item
        friend_item_type = items["Type"]       
        vector_index = t4_item_category_to_index_dict[(friend_item_type, like_item_name)]
        friend_vector_repr[vector_index] = 1
        
    
    return friend_vector_repr



t4_friend_likes_as_vectors = np.array(
                                [ t4d_convert_friend_likes_to_vector(friend_like_dtls) 
                                         for friend_like_dtls in  p4_fb_friend_like_dtls
                                         if len(friend_like_dtls["Likes"]) > 0 
                                ]
                             ) 



t4_indices_to_friend_names = [friend_like_dtls["Name"] 
                             for friend_like_dtls in  p4_fb_friend_like_dtls 
                             if len(friend_like_dtls["Likes"]) > 0]


# In[10]:

#Task t4e:

def t4e_cluster_friends(data, k):
    t4e_kmeansapp = KMeans(n_clusters=k, init='k-means++', random_state = 1234)
    t4e_kmeanspp_fit = t4e_kmeansapp.fit(data)
    return t4e_kmeanspp_fit




#Task 4f:

def t4f_plot_cluster_metrics(data):
    
    plt.figure()
    fig,axes = plt.subplots(1, 3, figsize=(15, 5))
    ks = range(2, 21)
    ssqs = ssq_statistics(data, ks=ks)
        ## create a line chart with x axis as different k values 
        #and y-axis as ssqs on axes[1] variable    
    axes[0].plot(ks,ssqs)
    axes[0].set_title("SSQ")
    axes[0].set_xlabel("$k$")
    axes[0].set_ylabel("SSQ")
    
    
    gaps, errs, difs = gap_statistics(data, nrefs=25, ks=ks)
    max_gap = None
    if len(np.where(difs > 0)[0]) > 0:
        max_gap = np.where(difs > 0)[0][0] + 1 # the k with the first positive dif
    if max_gap:
        print "By gap statistics, optimal k seems to be ", max_gap
    else:
        print "Please use some other metrics for finding k"
        
    #Create an errorbar plot
    rects = axes[1].errorbar(ks, gaps, yerr=errs, xerr=None, linewidth=1.0)

    #Add figure labels and ticks
    axes[1].set_title('Clustering Gap Statistics')
    axes[1].set_xlabel('Number of clusters k')
    axes[1].set_ylabel('Gap Statistic')
    axes[1].set_xticks(ks)
    # Add figure bounds
    axes[1].set_ylim(0, max(gaps+errs)*1.1)
    axes[1].set_xlim(0, len(gaps)+1.0)

    ind = range(1,len(difs)+1) # the x values for the difs
    
    max_gap = None
    if len(np.where(difs > 0)[0]) > 0:
        max_gap = np.where(difs > 0)[0][0] + 1 # the k with the first positive dif
    
    #Create a bar plot
    axes[2].bar(ind, difs, alpha=0.5, color='g', align='center')

    # Add figure labels and ticks
    if max_gap:
        axes[2].set_title('Clustering Gap Differences\n(k=%d Estimated as Optimal)' % (max_gap))
    else:
        axes[2].set_title('Clustering Gap Differences\n')
    axes[2].set_xlabel('Number of clusters k')
    axes[2].set_ylabel('Gap Difference')
    axes[2].xaxis.set_ticks(range(1,len(difs)+1))

    #Add figure bounds
    axes[2].set_ylim(min(difs)*1.2, max(difs)*1.2)
    axes[2].set_xlim(0, len(difs)+1.0)

t4f_plot_cluster_metrics(t4_friend_likes_as_vectors)


# In[72]:

#Task 4g

t4g_opt_k = 16
t4g_best_clusterings = t4e_cluster_friends(t4_friend_likes_as_vectors, t4g_opt_k )
print t4g_best_clusterings




#Task 4h

def t4h_print_cluster_with_friend_names(best_clusterings, indices_to_friend_names):
      
    clusters =  best_clusterings.labels_
    unique_clusters = np.unique(clusters)
    t4_names_cluster = []
    for i in unique_clusters:
        t4h_names_array= []
        for j in range (len(clusters)):
            if(clusters[j] == i):
                t4h_names_array.append(indices_to_friend_names[j])
        t4_names_cluster.append(t4h_names_array)
        print "Cluster %d:"%(i+1),t4_names_cluster[i]
 
    
t4h_print_cluster_with_friend_names(t4g_best_clusterings, t4_indices_to_friend_names)




#Task 4i
    find h friends who have the lowest distance to that cluster's centroid and print them

def t4i_print_top_representative_friends(best_clusterings, h=1):
    
    t4_cluster_dist =[]
    for i in range(len (best_clusterings.cluster_centers_)):
        t4h_dist_array = []
        for j in range(len(t4_indices_to_friend_names)):
            if (best_clusterings.labels_[j] == i):
                computed_distance = metrics.pairwise.euclidean_distances(best_clusterings.cluster_centers_[i], t4_friend_likes_as_vectors[j])
                friend_name = t4_indices_to_friend_names[j]
                t4h_dist_array.append((computed_distance,friend_name))
        print [Nearest_friends[1] for Nearest_friends in sorted(t4h_dist_array, key=lambda t4h_dist_array:  t4h_dist_array[0] )[:h]]                            
                
t4i_print_top_representative_friends(t4g_best_clusterings,5)





# # Part 5. Dimensionality Reduction
# 

mnist = fetch_mldata("MNIST original", data_home=".")       
             
                                                                                                                            
 

np.random.seed(1234)                        


p5_train_data, p5_test_data, p5_train_labels, p5_test_labels =         train_test_split(mnist.data, mnist.target, test_size=0.3)


p5_train_data = p5_train_data / 255.0                                        
p5_test_data = p5_test_data / 255.0


# In[5]:

#Task t5a:
# Plot the average value of all digits

plt.figure()
fig,axes = plt.subplots(2, 5, figsize=(15,4))

for i in range(10):
    t5a_row, t5a_col = i // 5, i%5
    ## Subset p5_train_data with images for digit i only 
    # Possible to do it 1 liner (similar to how it is done in Pandas)
    t5a_digit_i_subset = p5_train_data[p5_train_labels == i]
   

    ## compute avg value of t5a_training_data_sevens_only and t5a_training_data_nines_only 
    # remember to use a vectorized version of mean for efficiency
    t5a_digit_i_subset_mean = np.mean(t5a_digit_i_subset, axis = 0)

    #Do not #
    axes[t5a_row][t5a_col].imshow( t5a_digit_i_subset_mean.reshape(28, 28), cmap="Greys") 
    axes[t5a_row][t5a_col].grid(False)
    axes[t5a_row][t5a_col].get_xaxis().set_ticks([])
    axes[t5a_row][t5a_col].get_yaxis().set_ticks([])


# In[15]:

#Task t5b: train a multi class classifier (OneVsRest) with LinearSVC class and make predictions and print it. 

t5b_start_time = time.time()

## OvR classifier with LinearSVC class with default parameters and random state of 1234
t5b_mc_ovr_linear_svc_svm_model = OneVsRestClassifier(LinearSVC(random_state =1234))
## Train the model
t5b_mc_ovr_linear_svc_svm_model.fit(p5_train_data,p5_train_labels)
print "SVM training over all features took %s seconds" % (time.time() - t5b_start_time)

## Make predictions using the model
t5b_mc_ovr_predictions_linear_svm_svc = t5b_mc_ovr_linear_svc_svm_model.predict(p5_test_data)


print "SVM over all features has an accuracy score of %s" % (metrics.accuracy_score(p5_test_labels, t5b_mc_ovr_predictions_linear_svm_svc))


# In[7]:

#Task t5c

#Remember that MNIST images are 28x28 => 784 features.
#  Often the entire data is not needed and we can find interesting structure in lower dimensions
# Let us see how this works
#You might want to check http://scikit-learn.org/stable/modules/decomposition.html#decompositions for details


#Let us arbitrarily pick number of components as 100
t5c_start_time = time.time()
## instantiate PCA object with 100 components
t5c_pca = PCA(n_components=100)
t5c_pca.fit(p5_train_data)
## transform the training and test class data
t5c_train_data_pca = t5c_pca.transform(p5_train_data)
t5c_test_data_pca = t5c_pca.transform(p5_test_data)

print "PCA and transformation took %s seconds" % (time.time() - t5c_start_time)


t5c_start_time = time.time()
## OvR classifier with LinearSVC class with default parameters and random state of 1234
t5c_mc_ovr_linear_svc_svm_model = OneVsRestClassifier(LinearSVC(random_state =1234))
## Train the model using the TRANSFORMED training data
t5c_mc_ovr_linear_svc_svm_model.fit(t5c_train_data_pca, p5_train_labels)
print "SVM training over top-100 components took %s seconds" % (time.time() - t5c_start_time)

## Make predictions using the model over the TRANSFORMED testing data
t5c_mc_ovr_predictions_linear_svm_svc = t5c_mc_ovr_linear_svc_svm_model.predict(t5c_test_data_pca)



print "SVM over top-100 components has an accuracy score of %s" % (
    metrics.accuracy_score(p5_test_labels, t5c_mc_ovr_predictions_linear_svm_svc))


# In[8]:

#Task t5d: Heads up - This is a time consuming task 
# on my virtual machine with 4gb ram, it took approximately 30 minutes


#Means 1,2,3,4,5, 10, 20, 30, 40,, ... 200, 784
t5d_num_dimensions_to_test = list(reversed([1,2,3,4,5] + range(10, 200+1, 10) + [784]))

#Let us now see how varying number of components affects time and accuracy
t5d_columns = ["Num Components", "PCA Time", "Training Time", "Total Time", "Accuracy"]
t5d_results_df = DataFrame(0, index = t5d_num_dimensions_to_test, columns = t5d_columns)

for k in t5d_num_dimensions_to_test:
    print "Handling num dimensions = ", k
    t5d_start_time = time.time()
    
    ## instantiate PCA object with k components
    t5d_pca = PCA(n_components=k)
    
    t5d_pca.fit(p5_train_data)
    
    ## transform the training and testing class data
    t5d_train_data_pca = t5d_pca.transform(p5_train_data)
    t5d_test_data_pca = t5d_pca.transform(p5_test_data)
    
    t5d_pca_time = time.time() - t5d_start_time
    
    t5d_start_time = time.time()
    ## OvR classifier with LinearSVC class with default parameters and random state of 1234
    t5d_mc_ovr_linear_svc_svm_model = OneVsRestClassifier(LinearSVC(random_state =1234))
    ## Train the model using the TRANSFORMED training data
    t5d_mc_ovr_linear_svc_svm_model.fit(t5d_pca.transform(p5_train_data), p5_train_labels)
    
    t5d_training_time = time.time() - t5d_start_time
    
    
    ## Make predictions using the model over the TRANSFORMED testing data
    t5d_mc_ovr_predictions_linear_svm_svc = t5d_mc_ovr_linear_svc_svm_model.predict(t5d_test_data_pca)
    ## Compute the accuracy score
    t5d_accuracy = metrics.accuracy_score(p5_test_labels, t5d_mc_ovr_predictions_linear_svm_svc)

    #update df
    t5d_results_df.ix[k] = [k, t5d_pca_time, t5d_training_time, 
                                t5d_pca_time + t5d_training_time, t5d_accuracy]
    
display(t5d_results_df)


# In[9]:

#Task t5e


t5e_pca = t5c_pca



## using t5e_pca variable, print the cumulative variance that is explained
print "Total variance explained with 100 components is ", np.sum(t5e_pca.explained_variance_ratio_)

plt.figure()
fig,axes = plt.subplots(1, 2, figsize=(15,4))

# # plot the explained variance of these 100 components
axes[0].plot(t5e_pca.explained_variance_ratio_)
axes[0].set_title('Variance Explained by $i$-th Component')

# # plot the cumulative explained variance of these 100 components

axes[1].plot(np.cumsum(t5e_pca.explained_variance_ratio_))
axes[1].set_title('Variance Explained by top-$i$ Components')
plt.show()


# In[ ]:



