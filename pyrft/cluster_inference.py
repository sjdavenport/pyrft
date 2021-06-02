import numpy as np
import pyrft as pr
import math
from skimage import measure

def find_clusters(test_statistic, CDT, below = bool(0), mask = math.nan, connectivity = 1, two_sample = bool(0), minCsize = 1):
  """ find_clusters
  Parameters  
  ---------------------
  test_statistic:   a numpy.nd array,
          

  Returns
  ---------------------
  cluster_image:    a numpy.nd array,
              with the same size as the test-statistic in which the clusters 
              above the CDT are labelled each with a different number. 
  
  Examples
  ---------------------
# Clusters above 0.5
cluster_image, cluster_sizes = pr.find_clusters(np.array([[1,0,1],[1,1,0]]), 0.5)
# Clusters below 0.5
cluster_image, cluster_sizes = pr.find_clusters(np.array([[1,0,1],[1,1,0]]), 0.5, below = 1)
# tstat image
f = pr.statnoise((50,50), 20, 10)
tstat, xbar, std_dev = pr.mvtstat(f.field)
cluster_image, c_sizes = pr.find_clusters(tstat, 2)
plt.imshow(cluster_image)
  """
  
  # Mask the data if that is possible
  if np.sum(np.ravel(mask)) > 0:
      test_statistic = test_statistic*mask
      
  if two_sample:
      raise Exception("two sample hasn't been implemented yet!")
      
  if below:
     cluster_image = measure.label((test_statistic < CDT)*(test_statistic > 0), connectivity = connectivity)
  else:
     cluster_image = measure.label(test_statistic > CDT, connectivity = connectivity)
  
  n_clusters = np.max(cluster_image)
  store_cluster_sizes = np.zeros(1)
  
  # Use J to keep track of the clusters
  J = 0
  
  for I in np.arange(n_clusters):
      cluster_index = (cluster_image == (I+1))
      cluster_size = np.sum(cluster_index)
      if cluster_size < minCsize:
          cluster_image[cluster_index] = 0
      else:
          J = J + 1
          store_cluster_sizes = np.append(store_cluster_sizes, cluster_size)
          cluster_image[cluster_index] = J
          
  # Remove the initial zero
  store_cluster_sizes = store_cluster_sizes[1:]
  
  return cluster_image, store_cluster_sizes
  