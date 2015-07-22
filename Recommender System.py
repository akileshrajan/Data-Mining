
# coding: utf-8
# ## Name: Akilesh R. Student ID.1001091662
# ## Name: Ashwin R. Student ID.1001098716 
# ## Name: Anirudh R. Student ID.1001051262
# ## Code Implementation based on Lecture from Saravanan Thirumuruganathan, University of Texas at Arlington
# 
########### Do not change anything below

get_ipython().magic(u'matplotlib inline')

#Array processing
import numpy as np

#Data analysis, wrangling and common exploratory operations
import pandas as pd
from pandas import Series, DataFrame
from IPython.display import display

#For visualization. Matplotlib for basic viz and seaborn for more stylish figures + statistical figures not in MPL.
import matplotlib.pyplot as plt
import seaborn as sns


import scipy as sp
#SVD for Sparse matrices
from scipy.sparse.linalg import svds

from sklearn.metrics.pairwise import euclidean_distances

try:
   import cPickle as pickle
except:
   import pickle

from collections import defaultdict, Counter
import operator

from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 6)


import itertools
import csv
#######################End imports###################################


# # Part 1: Exploratory Analysis/Sparsity


#####Do not change anything below

#Load the user data
users_df = pd.read_csv('ml-100k/u.user', sep='|', names=['UserId', 'Age', 'Gender', 'Occupation', 'ZipCode'])
#Load the movies data: we will only use movie id and title for this assignment
movies_df = pd.read_csv('ml-100k/u.item', sep='|', names=['MovieId', 'Title'], usecols=range(2))
#Load the ratings data: ignore the timestamps
ratings_df = pd.read_csv('ml-100k/u.data', sep='\t', names=['UserId', 'MovieId', 'Rating'],usecols=range(3))

#Working on three different data frames is a pain
# Let us create a single dataset by "joining" these three data frames
movie_ratings_df = pd.merge(movies_df, ratings_df)
movielens_df = pd.merge(movie_ratings_df, users_df)

ratings_df .head()


# In[3]:

#Task t1a: Print the NAME of the top-10 movies with most ratings

print movie_ratings_df['Title'].groupby(movie_ratings_df.Title).count(movie_ratings_df.UserId).order(ascending = False)[:10]


# In[4]:

#Task t1b: 


#t1b_user_rating_count is a groupby object that counts the number of ratings for each user.
t1b_user_rating_count = movielens_df['UserId'].groupby(movielens_df.UserId).count().hist(bins = 50)
t1b_user_rating_count.set_xlabel('Ratings per user')
t1b_user_rating_count.set_ylabel('#Users')
t1b_user_rating_count.set_title('Count of Ratings per User')
plt.figure()




# In[5]:

#Task t1c: 
# Title="Count of Ratings per Movie", XLabel="Ratings per Movie", YLabel="#Movies"

#t1c_user_rating_count is a groupby object that counts the number of ratings for each movie.
t1c_user_rating_count =movie_ratings_df['MovieId'].groupby(movie_ratings_df.MovieId).count(movie_ratings_df.Rating).hist(bins = 50)
t1c_user_rating_count.set_xlabel('Ratings per Movie')
t1c_user_rating_count.set_ylabel('#Movies')
t1c_user_rating_count.set_title('Count of Ratings per Movie')
plt.figure()
####The figure below shows that most movies receive less than 25 ratings while few popular get a lot of ratings


# In[6]:

#Task t1d: Let us now analyze the rating distribution
#  Create a histogram based on the ratings received for each movie with 5 bins. 
# Title="Ratings Histogram", XLabel="Rating Provided", YLabel="#Ratings"




t1d_user_rating_count = movie_ratings_df['Rating'].hist(bins = 5)
t1d_user_rating_count.set_xlabel('Rating Provided')
t1d_user_rating_count.set_ylabel('#Ratings')
t1d_user_rating_count.set_title('Ratings Histogram')
plt.figure()
print "Average rating of ALL movies is", movie_ratings_df['Rating'].mean().round(decimals=2)


# In[7]:

#Task t1e:
t1e_avg_ratings = movie_ratings_df['Rating'].groupby(movie_ratings_df.UserId).mean().hist(bins = 5)
t1e_avg_ratings.set_xlabel('Average Rating')
t1e_avg_ratings.set_ylabel('#Users')
t1e_avg_ratings.set_title('Histogram of Average Ratings of Users')
plt.figure()



# In[151]:


t1f_avg_ratings = movie_ratings_df['Rating'].groupby(movie_ratings_df.MovieId).mean().hist(bins = 5)
t1f_avg_ratings.set_xlabel('Average Rating')
t1f_avg_ratings.set_ylabel('#Movies')
t1f_avg_ratings.set_title('Histogram of Average Ratings of Movies')



#Task t1g:

t1g_all_movies = movies_df['MovieId'].unique()

t1g_allpair_commonsupport = []


for mov1,mov2 in itertools.combinations(t1g_all_movies, 2):
    mov1_reviewers = movie_ratings_df[movie_ratings_df.MovieId==mov1].UserId.unique()
    mov2_reviewers = movie_ratings_df[movie_ratings_df.MovieId==mov2].UserId.unique()
    commonsupport = set(mov1_reviewers).intersection(mov2_reviewers)
    t1g_allpair_commonsupport.append(len(commonsupport))
            
print "Average common support is ", round(np.mean(t1g_allpair_commonsupport), 2)
plt.hist(t1g_allpair_commonsupport)




#Task t1h: 

t1h_sparsity = float(len(movie_ratings_df['Rating']))/(float(len(movie_ratings_df['UserId'].unique()))*float(len(movie_ratings_df['MovieId'].unique())))
print "Sparsity of the dataset is ", t1h_sparsity



#Task t1i: 


t1i_movielens_mean_ratings = movielens_df.pivot_table(values='Rating',index='Title',columns = 'Gender',aggfunc=np.mean)
display(t1i_movielens_mean_ratings[:10])


# # Part 2: Nearest Neighbor based Recommender System


#Task t2a: 

movie_name_to_id_dictionary = {}
movie_name_to_id_dictionary = dict(zip(movies_df.Title,movies_df.MovieId))

all_movie_names = []
#Your code below
all_movie_names = list(movies_df.Title)


# In[13]:

#Task t2b: Write a function that takes two inputs: 
#  movie_id: id of the movie and common_users: a set of user ids
# and returns the list of rows corresponding to their movie ratings 

def get_movie_reviews(movie_id, common_users):
    #Get a boolean vector for themovielens_dfns_dfs provided by users in common_users for movie movie_id
    # Hint: use the isin operator of Pandas
    #mask = movielens_df['Rating'][movielens_df.MovieId == movie_id][movielens_df.UserId].isin(common_users)
    mask = movielens_df.UserId.isin(common_users)
    #Create a subset of data where the mask is True i.e. only collect data from users who satisfy the condition above
    # Then sort them based on userid
    Value_intersection = (mask) & (movielens_df.MovieId==movie_id)
    movie_ratings = movielens_df[Value_intersection].sort(columns = 'UserId', ascending = True)
    
    #Do not change below
    #Return the unique set of ratings provided
    movie_ratings = movie_ratings[movie_ratings['UserId'].duplicated()==False]
    return movie_ratings
#common_users = [1, 5, 521, 13, 532, 536, 42, 561, 64, 72, 83, 92, 95, 102, 618, 621, 622, 632, 642, 643, 193, 648, 650, 653, 655, 660, 682, 178, 705, 709, 200, 201, 715, 213, 727, 292, 222, 738, 746, 749, 751, 757, 249, 250, 764, 256, 773, 268, 130, 271, 387, 276, 790, 279, 280, 795, 796, 798, 804, 293, 806, 807, 301, 815, 305, 826, 830, 320, 325, 327, 864, 868, 870, 363, 880, 886, 889, 378, 379, 892, 234, 896, 899, 393, 398, 399, 303, 916, 407, 924, 416, 934, 425, 429, 435, 450, 49, 455, 472, 484, 374, 487, 495, 497]  
#get_movie_reviews(100,set(range(1,10)))


# In[14]:

#Do not change below

#Here are some sample test cases for evaluating t2b
print "get_movie_reviews(1, set([1]))"
display( get_movie_reviews(1, set([1])) )

print "get_movie_reviews(1, set(range(1, 10)))"
display( get_movie_reviews(1, set(range(1, 10))) )

print "get_movie_reviews(100, set(range(1, 10)))"
display( get_movie_reviews(100, set(range(1, 10))) )

print "get_movie_reviews(784, set(range(1, 784)))"
display( get_movie_reviews(784, set(range(1, 784))) )




# In[99]:

#Task t2c: 

def calculate_similarity(movie_name_1, movie_name_2, min_common_users=0):
    
    movie1 = movie_name_to_id_dictionary[movie_name_1]
    movie2 = movie_name_to_id_dictionary[movie_name_2]
    
    #This is the set of UNIQUE user ids  who reviewed  movie1
    users_who_rated_movie1 = movielens_df[movielens_df.MovieId==movie1].UserId.unique()
     
    #This is the set of UNIQUE user ids  who reviewed  movie2
    users_who_rated_movie2 = movielens_df[movielens_df.MovieId==movie2].UserId.unique()
    
    #Compute the common users who rated both movies: 
    # hint convert both to set and do the intersection
    common_users = set(users_who_rated_movie1).intersection(set(users_who_rated_movie2))
    
    #Using the code wrote in t2b, get the reviews for the movies and common users
    movie1_reviews = get_movie_reviews(movie1, common_users)
    movie2_reviews = get_movie_reviews(movie2, common_users)
    
    
    
    distance = euclidean_distances(movie1_reviews.Rating, movie2_reviews.Rating)

    if len(common_users) < min_common_users:
        return [[float('inf')]]
    return distance



#Do not change below
print calculate_similarity("Toy Story (1995)", "GoldenEye (1995)")
print calculate_similarity("GoldenEye (1995)", "Tomorrow Never Dies (1997)")
print calculate_similarity("Batman Forever (1995)", "Batman & Robin (1997)")


# In[101]:

#Task t2d: 

def get_top_k_similar_movies(input_movie_name, k=5, min_common_users=0):
    Similarity_list = []
    
    for movies in all_movie_names:
        if(movies != input_movie_name):
            Similarity = float(calculate_similarity(input_movie_name, movies,min_common_users))
            Similarity_list.append((Similarity,movies))
    return sorted(Similarity_list)[:k]


# In[18]:

#print get_top_k_similar_movies("Toy Story (1995)", 10)
print "\nMovies similar to GoldenEye [25]", get_top_k_similar_movies("GoldenEye (1995)", 10, 25)
print "\nMovies similar to GoldenEye [50]", get_top_k_similar_movies("GoldenEye (1995)", 10, 50)
print "\nMovies similar to GoldenEye [100]", get_top_k_similar_movies("GoldenEye (1995)", 10, 100)
print "\n\n"

print "\nMovies similar to Usual Suspects [25]", get_top_k_similar_movies("Usual Suspects, The (1995)", 10, 25)
print "\nMovies similar to Usual Suspects [50]", get_top_k_similar_movies("Usual Suspects, The (1995)", 10, 50)
print "\nMovies similar to Usual Suspects [100]", get_top_k_similar_movies("Usual Suspects, The (1995)", 10, 100)
print "\n\n"

print "\nMovies similar to Batman Forever [25]", get_top_k_similar_movies("Batman Forever (1995)", 10, 25)
print "\nMovies similar to Batman Forever [50]", get_top_k_similar_movies("Batman Forever (1995)", 10, 50)
print "\nMovies similar to Batman Forever [100]", get_top_k_similar_movies("Batman Forever (1995)", 10, 100)
print "\n\n"

print "\nMovies similar to Shawshank Redemption [25]", get_top_k_similar_movies("Shawshank Redemption, The (1994)", 10, 25)
print "\nMovies similar to Shawshank Redemption [50]", get_top_k_similar_movies("Shawshank Redemption, The (1994)", 10, 50)
print "\nMovies similar to Shawshank Redemption [100]", get_top_k_similar_movies("Shawshank Redemption, The (1994)", 10, 100)
print "\n\n"


# #Task 3: Item based Collaborative Filtering
# 

#Do not change below
def euclidean_distance_normed(vec1, vec2):
    if len(vec1) == 0:
        return 0.0
    euc_distance = euclidean_distances(vec1, vec2)[0][0]
    return 1.0 / (1.0 + euc_distance)


#Task t3a:
def calculate_similarity_normed(movie_name_1, movie_name_2, min_common_users=0):
    movie1 = movie_name_to_id_dictionary[movie_name_1]
    movie2 = movie_name_to_id_dictionary[movie_name_2]

    #This is the set of UNIQUE user ids  who reviewed  movie1
    users_who_rated_movie1 = movielens_df[movielens_df.MovieId==movie1].UserId.unique()
    
    #This is the set of UNIQUE user ids  who reviewed  movie2
    users_who_rated_movie2 = movielens_df[movielens_df.MovieId==movie2].UserId.unique()
     


    common_users = set(users_who_rated_movie1).intersection(set(users_who_rated_movie2))


    movie1_reviews = get_movie_reviews(movie1, common_users)
    movie2_reviews = get_movie_reviews(movie2, common_users)
    #Do not change below
    
   
    distance = euclidean_distance_normed(movie1_reviews['Rating'].values, movie2_reviews['Rating'].values)

    if len(common_users) < min_common_users:
        return 0.0
    return distance




#Do not change below
print calculate_similarity_normed("Toy Story (1995)", "GoldenEye (1995)")
print calculate_similarity_normed("GoldenEye (1995)", "Tomorrow Never Dies (1997)")
print calculate_similarity_normed("Batman Forever (1995)", "Batman & Robin (1997)")




movie_similarity_hash = defaultdict(dict)





#Task t3b: 
# Get the top-k movie names with most ratings


def top_k_movie_names(k):
    movie_ratings_counter = Counter()
    
    
    movie_ratings_counter.update(movie_ratings_df['Title'])
    return movie_ratings_counter.most_common(k)


print "Top-10\n", top_k_movie_names(10), "\n"
print "Top-25\n", top_k_movie_names(25), "\n"


# In[35]:

#Do not change below
top_250_movie_names = [item[0] for item in top_k_movie_names(250)]




#Task t3c:

def compute_movie_to_movie_similarity(movie_names, min_common_users=0):
    #Your code below
    for mov1,mov2 in itertools.combinations(movie_names, 2):
        sim_val = calculate_similarity_normed(mov1,mov2,min_common_users)
        movie_similarity_hash[mov1][mov2]= sim_val
        movie_similarity_hash[mov2][mov1]= sim_val
    


# In[95]:


movie_similarity_hash = defaultdict(dict)


compute_movie_to_movie_similarity(top_250_movie_names[:10], min_common_users=0)

display(movie_similarity_hash["Toy Story (1995)"])
display(movie_similarity_hash['Return of the Jedi (1983)'])

print movie_similarity_hash["Toy Story (1995)"]["Independence Day (ID4) (1996)"]


movie_similarity_hash = defaultdict(dict)
compute_movie_to_movie_similarity(top_250_movie_names, min_common_users=25)


for movie_name in top_250_movie_names[:10]:
    print "Top-10 most similar movies for ", movie_name, " :", 
    print sorted(movie_similarity_hash[movie_name].items(), key=operator.itemgetter(1), reverse=True)[:10]
    print "\n"



#Task t3d
def predict_rating_for_movie_icf(movie_similarity_hash, input_user_id, input_movie_name, movies_considered):
    total_weighted_rating = 0.0
    total_similarity= 0.0
        
    user_movie_rating = user_rating_hash[input_user_id]   
    if input_movie_name not in user_movie_rating:
        
        for movies in user_movie_rating.keys():
            if movies in movies_considered:
                movie_sim = movie_similarity_hash[movies][input_movie_name]
                if movie_sim != 0:
                    total_similarity += movie_sim
                    weighted_rating = movie_sim * user_movie_rating[movies]
                    total_weighted_rating += weighted_rating
        #Do not change below
        #print total_similarity, total_weighted_rating
        if total_similarity == 0.0:
            return 0.0
        return total_weighted_rating / total_similarity
    else:
        return user_movie_rating[input_movie_name]



for user_id in range(1, 5+1):
    
    print user_id, [ (movie_name, 
                        round(predict_rating_for_movie_icf(movie_similarity_hash, user_id, movie_name, top_250_movie_names),2))
                       for movie_name in top_250_movie_names[:20] 
                        if movie_name not in user_rating_hash[user_id]]
           

def recommend_movies_icf(input_user_id, movies_considered, movie_similarity_hash,
                             user_rating_hash, k=10, min_common_movies=5):
    predicted_ratings = []
    
    #Your code here
    for movies in movies_considered:
        if movies not in user_rating_hash[input_user_id]:
            predicted_ratings.append((
                predict_rating_for_movie_icf(movie_similarity_hash, input_user_id, movies, top_250_movie_names), movies))

    
    return sorted(predicted_ratings, reverse=True)[:k]


# In[155]:

#Do not change below:

#Let us predict top-5 movies for first 10 users
for user_id in range(1,11):
    print user_id, recommend_movies_icf(user_id, top_250_movie_names, movie_similarity_hash, 
                               user_rating_hash, k=10, min_common_movies=5)


# #Task 4: User based Collaborative Filtering


#Task t4a


def compute_user_rating_hash():
    user_rating_hash = defaultdict(dict)
    
    
    with open('ml-100k/u.data') as UUIDfile:
        reader = csv.reader(UUIDfile,delimiter='\t')
        
        for row in reader:
            user_rating_hash[int(row[0])][all_movie_names[int(row[1])-1]]=int(row[2])
    return user_rating_hash


# In[65]:


user_rating_hash = compute_user_rating_hash()


print len(user_rating_hash.keys())
#How many movies did each of the first 20 users rated?
print [len(user_rating_hash[i].keys()) for i in range(1,20+1)] 
#print the ratings of user 4
display(user_rating_hash[4])



def compute_user_user_similarity(user_rating_hash, user_id_1, user_id_2, min_common_movies=0):
    
    user1_names = (user_rating_hash[user_id_1].keys())
    user2_names = (user_rating_hash[user_id_2].keys())
    
    #compute common movies
    common_movies = set(user1_names) & set(user2_names)
    
    if len(common_movies) < min_common_movies:
        return 0.0
    
    common_movies = sorted(list(common_movies))
    #vector1 is the set of ratings for user1 for movies in common_movies
    vector1 = []
    for movies in common_movies:
        vector1.append(user_rating_hash[user_id_1][movies])
    #vector2 is the set of ratings for user2 for movies in common_movies
    vector2 = []
    for movies in common_movies:
        vector2.append(user_rating_hash[user_id_2][movies])
    
    #Compute distance and return 1.0/(1.0+distance)
    distance = euclidean_distances(vector1, vector2)[0][0]
    return 1.0 / ( 1.0 + distance)

#Testing code
print [round(compute_user_user_similarity(user_rating_hash, 1, i),2) for i in range(1, 10+1)]
print [round(compute_user_user_similarity(user_rating_hash, 784, i),2) for i in range(1, 10+1)]


#Task t4c

def top_k_most_similar_users(user_rating_hash, input_user_id, all_user_ids, k=10, min_common_movies=0):
    user_similarity = []
        
    #Your code below
    for users in all_user_ids:
        if (users != input_user_id):
            user_user_similarity = compute_user_user_similarity(user_rating_hash, input_user_id, users, min_common_movies)
            user_similarity.append((user_user_similarity,users))      

    return sorted(user_similarity, reverse=True)[:k]


# In[146]:

#Do not change below
all_user_ids = range(1, 943+1)
print top_k_most_similar_users(user_rating_hash, 1, all_user_ids, 10, 5)
print top_k_most_similar_users(user_rating_hash, 1, all_user_ids, 10, 10)
print top_k_most_similar_users(user_rating_hash, 812, all_user_ids, 10, 5)
print top_k_most_similar_users(user_rating_hash, 812, all_user_ids, 10, 20)


# In[149]:

#Task t4d

def predict_rating_for_movie_ucf(user_rating_hash, input_user_id, movie_name, all_user_ids, min_common_movies=5):
    total_weighted_rating = 0.0
    total_similarity= 0.0

    #For each user id
    for users in all_user_ids:
        if (users != input_user_id):
        
        #compute similarity between users
            user_similarity = compute_user_user_similarity(user_rating_hash, input_user_id, users, min_common_movies)
        
        
            if (user_similarity != 0.0):
                if (movie_name in user_rating_hash[users]): 
                    weighted_rating = user_rating_hash[users][movie_name]*user_similarity
                    total_weighted_rating +=  weighted_rating
                    total_similarity += user_similarity
        
    
    if total_similarity == 0.0:
        return 0.0
    
    return total_weighted_rating / total_similarity


# In[150]:

#Do not change below
all_user_ids = range(1, 943+1)
for user_id in range(1, 5+1):
    print "user_id = ", user_id
    print [ round(predict_rating_for_movie_ucf(user_rating_hash, user_id, all_movie_names[i], all_user_ids, min_common_movies=5),1)
          for i in range(1, 10+1)]
    print [ round(predict_rating_for_movie_ucf(user_rating_hash, user_id, all_movie_names[i], all_user_ids, min_common_movies=10),1)
          for i in range(1, 10+1)]
    print "\n"


# In[163]:

#Task t4e: 

def recommend_movies_ucf(user_rating_hash, all_user_ids, input_user_id, k=10, min_common_movies=5):
    predicted_ratings = []
    
    #Your code here
    for movies in all_movie_names:
        if movies not in user_rating_hash[input_user_id]:
            predicted_ratings.append(
                (predict_rating_for_movie_ucf(user_rating_hash, input_user_id, movies, all_user_ids, min_common_movies),movies)) 
    return sorted(predicted_ratings, reverse=True)[:k]


# In[165]:

#Do not change below
all_user_ids = range(1, 943+1)

for user_id in range(1, 5):
    
    print recommend_movies_ucf(user_rating_hash, all_user_ids, user_id, k=10, min_common_movies=5)


# #Task 5: Latent Factor Models
# 
# In this task, let us try to find the simplest SVD based latent factor model.

# In[70]:

number_of_users = 943
number_of_movies = 1682

ratings_matrix = sp.sparse.lil_matrix((number_of_users, number_of_movies))



# In[75]:

#Task t5a: 

with open("ml-100k/u.data","r") as f:
    for line in f:
        user_id, movie_id, rating, timestamp = line.split("\t")
        ratings_matrix[ int(user_id)-1, int(movie_id)-1] = int(rating)


# In[79]:

print "Matrix shape is", ratings_matrix.shape
print "Number of non zero values", ratings_matrix.nnz
print "Number of non zero values", ratings_matrix.nonzero()





