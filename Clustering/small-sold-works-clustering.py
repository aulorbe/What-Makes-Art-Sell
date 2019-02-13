
# coding: utf-8

# In[6]:


import pandas as pd

sold_works_df = pd.read_csv('/Users/flatironschool/Final-Project/final-project/sold_works_df')


# In[4]:


sold_works_df


# # Clustering images per size

# ## Clustering Small Images

# In[7]:


small_works = sold_works_df[sold_works_df['Size of Piece'] == 'Small']


# In[9]:


small_works.shape


# In[20]:


small_works['File Name']


# ### Getting RGB vals for every "small" image

# In[31]:


import os, os.path
import cv2
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import keras
import shutil


# In[39]:


# Putting 'small' sold images into folder called "small-sold-images"

os.mkdir('/Users/flatironschool/Final-Project/small-sold-works')

for file in os.listdir('/Users/flatironschool/Final-Project/All-Images'):
    for i in small_works['File Name']:
        if file == i: 
            shutil.copy(f'/Users/flatironschool/Final-Project/All-Images/{file}',f'/Users/flatironschool/Final-Project/small-sold-works/{i}')


# In[40]:


in_small_works_folder = []

for i in os.listdir('/Users/flatironschool/Final-Project/small-sold-works'): 
    in_small_works_folder.append(i)


# In[226]:


len(in_small_works_folder)


# In[56]:


# Checking that I successfully captured every file I wanted to & put it into the small-sold-images folder

list(set(small_works['File Name']) - set(in_small_works_folder))


# In[222]:


# Getting RGB vals for all images in small-sold-images folder

images = [] 
    
for file in os.listdir('/Users/flatironschool/Final-Project/small-sold-works'):
            
        #   Read the image
            image = cv2.imread(f'/Users/flatironschool/Final-Project/small-sold-works/{file}')

        #   Resize it
            image = cv2.resize(image,(224,224))

        #   Now we add it to our array
            images.append(image)


# In[61]:


# Normalizing RGB vals to make them easier to interpret

# def normalise_images(images):

#     # Convert to numpy arrays
#     images = np.array(images, dtype=np.float32)

#     # Normalise the images
#     images /= 255
    
#     return images


# In[62]:


# images_normalized = normalise_images(images)


# In[218]:


# images_normalized.shape


# In[65]:


# Resizing to a 2D array in order to feed to k-means

# images_reshaped = images_normalized.reshape(images_normalized.shape[0], -1)


# In[217]:


# images_reshaped.shape


# In[103]:


images_arr = np.array(images)

images_orig_2d = images_arr.reshape(images_arr.shape[0], -1)


# In[169]:


images_orig_2d


# ### Finding ideal k-value

# In[316]:


# Running PCA to reduce feature dimensionality

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaling', StandardScaler()), 
    ('pca', PCA(n_components = 3)),
    ('clust', KMeans(n_clusters = 3, max_iter=100, n_init=1, verbose=1)) # changed n_clusters to 2 after doing elbow & sil. plots
])

pipeline_fit = pipeline.fit_transform(images_orig_2d)
pipeline_output = pipeline.predict(images_orig_2d)


# In[125]:


set(pipeline_output)


# In[317]:


# Graphing 

from mpl_toolkits.mplot3d import Axes3D

cluster_centers = pipeline.named_steps['clust'].cluster_centers_
preds = pipeline_output

fig = plt.figure(figsize=(10,8))
ax = Axes3D(fig)
ax.scatter(pca_transformed[:, 0], pca_transformed[:, 1],pca_transformed[:,2], c=preds, depthshade=True)
# ax.scatter(x1,y1,z1 ,marker='.', c='red', s=500)
# ax.scatter(x2,y2,z2,marker='.',c='blue',s=500)


# In[206]:


# cluster_centers


# In[207]:


# cluster_centers[0,:]


# In[208]:


# x1,y1,z1 = cluster_centers[0,:]
# x2,y2,z2 = cluster_centers[1,:]


# In[182]:


# For graphing purposes because strangely pipeline is transforming the clusters, not the original data

pca_transformed = pipeline.named_steps['pca'].transform(images_orig_2d)


# In[107]:


# Using elbow method to find optimal value for k

from sklearn import metrics
from scipy.spatial.distance import cdist

distortions = []
K = range(1,10)

for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(pca_images)
    distortions.append(sum(np.min(cdist(pca_images, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / pca_images.shape[0])


# In[108]:


# Plotting

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# In[109]:


# Using silhouette method to find optimal value for k

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn import datasets

silhouette_plot = []

K = range(2,10) # need to switch from 1-10 to 2-10 because Silhouette score doesn't work with only 1 cluster (because then all labels = 0)

for k in K:
    clusters = KMeans(n_clusters=k, random_state=10)
    cluster_labels = clusters.fit_predict(pca_images)
    silhouette_avg = metrics.silhouette_score(pca_images, cluster_labels)
    silhouette_plot.append(silhouette_avg)


# In[110]:


plt.figure(figsize=(15,8))
plt.subplot(121, title='Silhouette coefficients over k')
plt.xlabel('k')
plt.ylabel('silhouette coefficient')
plt.plot(range(2, 10), silhouette_plot)
plt.axhline(y=np.mean(silhouette_plot), color="red", linestyle="--")
plt.grid(True)


# ## Matching k-means predictions back to images to inspect clusters

# In[318]:


preds


# In[224]:


images_orig_2d.shape


# In[228]:


len(in_small_works_folder)


# In[319]:


small_works_and_preds = list(zip(in_small_works_folder,preds))


# In[320]:


small_works_and_preds


# In[321]:


small_preds = []

for i in small_works['File Name']:
#     print(i)
    for y in small_works_and_preds:
#         print(y[0])
        if i == y[0]:
            small_preds.append(y[1])
            
        


# In[322]:


small_works['Clusters'] = small_preds


# In[323]:


small_works.head()


# ## Inspecting 3 clusters for small pieces

# In[324]:


cluster_0_images = small_works['File Name'][small_works['Clusters'] == 0]
cluster_1_images = small_works['File Name'][small_works['Clusters'] == 1]
cluster_2_images = small_works['File Name'][small_works['Clusters'] == 2]


# In[325]:


len(cluster_0_images)


# In[326]:


len(cluster_1_images)


# In[327]:


len(cluster_2_images)


# In[349]:


import matplotlib.image as mpimg

def show_images(cluster_df):
    for file in cluster_df:
        path = '/Users/flatironschool/Final-Project/small-sold-works/'
        read = mpimg.imread(path+file)
        plt.imshow(read)
        plt.show()
    


# In[350]:


show_images(cluster_1_images)


# In[330]:


show_images(cluster_0_images)


# In[331]:


show_images(cluster_2_images)


# In[389]:


# Seeing different clusters for a specific artist

def show_images_clusters_artist(artist_df):
    
    artist_df = artist_df.sort_values(['Clusters'])
    
    for i in range(0,len(artist_df)):
        path = '/Users/flatironschool/Final-Project/small-sold-works/'
        file = artist_df['File Name'].iloc[i]
        read = mpimg.imread(path+file)
        plt.imshow(read)
        plt.title(artist_df['Clusters'].iloc[i])
        plt.show()
    


# In[390]:


chong_df = small_works[small_works['Artist'] == 'Cecile Chong']

show_images_clusters_artist(chong_df)


# ## Clustering Medium Images

# In[10]:


med_works = sold_works_df[sold_works_df['Size of Piece'] == 'Medium']


# In[12]:


med_works.shape


# ## Clustering Large Image

# In[11]:


large_works = sold_works_df[sold_works_df['Size of Piece'] == 'Large']


# In[13]:


large_works.shape


# # Clustering images per type

# # Clustering images by season sold
