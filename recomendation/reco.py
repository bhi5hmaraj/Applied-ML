# A dictionary of movie critics and their ratings of a small
# set of movies

from math import sqrt
import numpy as np


# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2):

  # if they have no ratings in common, return 0
  p1 = prefs[person1]
  p2 = prefs[person2]
  both = p1 * p2
  if both is np.zeros_like(p1):
    return 0
    
      # Add up the squares of all the differences
  sum_of_squares = sum(np.power(p1 - p2, 2)[both > 0])

  return 1/(1+sum_of_squares)

# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(person1,person2):
  # Get the list of mutually rated items
  p1 = prefs[person1]
  p2 = prefs[person2]
  both = (p1 * p2)
  n=len(both[both >0])
  if n == 0:
    return 0

  # Sum calculations
  
  # Sums of all the preferences
  sum1 = sum(p1[both > 0])
  sum2 = sum(p2[both > 0])
  
  # Sums of the squares
  sum1Sq=sum(np.power(p1[both > 0], 2))
  sum2Sq = sum(np.power(p2[both > 0], 2))
  
  # Sum of the products
  pSum=sum(both)
  
  # Calculate r (Pearson score)
  num=pSum-(sum1*sum2/n)
  den=sqrt((sum1Sq-pow(sum1,2)/n)*(sum2Sq-pow(sum2,2)/n))
  if den==0: return 0

  r=num/den

  return r

# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(person,similarity=sim_pearson):
  
    sim = np.zeros(len(user_enum))

    for i in range(len(user_enum)):
        if i == person: continue
        s = similarity(person, i)
        if s <= 0: continue

        sim[i] = s


    weighted_sum = sim.dot(prefs) # 1 x n * n x m 
    # ignore scores of zero or lower
    sim_sum = np.zeros(len(movies))
    for i in range(len(sim_sum)):
        sim_sum[i] = np.sum(sim[prefs[:,i] > 0])

    score = np.divide(weighted_sum, sim_sum)
    rankings = [(movies[i], score[i]) for i in range(len(movies))]

    rankings.sort(key= lambda x : -x[1])
    return rankings


movies = []
movies_enum = {}

user_enum   = {}


def loadMovieLens(path='./ml-latest-small/'):
# Get movie titles

    for line in open(path+'movies.csv'):
        (id,title)=line.split(',')[0:2]
        movies_enum[id] = len(movies_enum)
        movies.append(title)

    # Load data


    for line in open(path+'ratings.csv'):
        (user,movieid,rating,ts)=line.split(',')
        if user in user_enum : continue

        user_enum[user] = len(user_enum)

    p = np.zeros((len(user_enum), len(movies)))

    for line in open(path+'ratings.csv'):
        (user,movieid,rating,ts)=line.split(',')
        
        user_pos = user_enum[user]
        p[user_pos][movies_enum[movieid]] = float(rating)
    

    return p


prefs = loadMovieLens()
print(getRecommendations(1)[:20])