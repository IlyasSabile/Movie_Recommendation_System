# Movie_Recommendation_System

This repository contains the two recommender systems that are mostly used in industry. The first one predicts a binary outcome built with Restricted Boltzman Machine. Indeed, it predicts whether a user is going to like yes or no a movie. And the second one is built by an AutoEncoder and predicts the rating from 1 to 5 of a movie by a user.

# Datasets 
The datasets used to trained the models are extracted from the most real world dataset which is the MovieLens Dataset.
https://grouplens.org/datasets/movielens/

GroupLens Research has collected and made available rating data sets from the MovieLens web site (http://movielens.org). The data sets were collected over various periods of time, depending on the size of the set. 

  - MovieLens 100K Dataset:
Stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies. Released 4/1998.
  - MovieLens 1M Dataset
Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003.

Before using these data sets, please review their README files for the usage licenses and other details.

# Dependencies
Python3, Pytorch, numpy, pandas.

These codes are implemented using the Deep Learning framework Pytorch that you can install from http://pytorch.org
or Run these Commands: 
- For Linux users: pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl 
                   pip3 install torchvision
- For OSX users: pip3 install http://download.pytorch.org/whl/torch-0.3.1-cp36-cp36m-macosx_10_7_x86_64.whl 
                 pip3 install torchvision


# Restricted Boltzman Machine 
I built a Recommandation system that predict a binary ratings outcome (1 if the users liked the movie and 0 if not) by training a RBM on the MovieLens dataset that contains 100k ratings from 1000 users on 1700 movies.

Approximating the RBM log-Likelihood Gradient
All common training algorithms for RBMs approximate the log-likelihood gradient given some data and perform gradient ascent on these approximations. 

I computed a K-steps Contrastive Divergence Algorithm to approximate the gradient where a Gibbs chain is run for only k steps (and usually k = 1). The Gibbs chain is initialized with a training example v(0) of the training set and yields the sample v(k) after k steps. Each step t consists of sampling h(t) from p(h|v(t)) and sampling v(t+1) from p(v|h(t)) subsequently. 

The Algorithm for k-step contrastive divergence is the following: 

<img width="620" alt="screen shot 2018-04-01 at 5 59 00 pm" src="https://user-images.githubusercontent.com/34433140/38177987-fb5c387e-35d6-11e8-9b05-49f0f90bc718.png">

- Results

To evaluate the rbm model results I computed the Average Distcance and I end up with a average distance of 0.24 which is equivalent to a 75% of correct predictions. 

Train/Test loss = the mean of the absolute distances between the predictions and the ratings.

Train Loss = 0.25 which means that we have 75% of succes.

Test Loss = 0.24 

# AutoEncoder

This time the recommender system is predicting the movies ratings from 1 to 5 by training an AutoEncoder on the same MovieLens dataset.  

I created a classe that inheritates from an existant parent class in pytorch called Module. I also used the RMSprop optimizer to reduce the mean square error at each epoch. 

- Results 

Train Loss  = 0.91

Test Loss = 0.95
