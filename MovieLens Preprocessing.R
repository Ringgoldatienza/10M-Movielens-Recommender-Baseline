#Preprocessing

#histogram
hist(edxTestSet$rating, breaks = 10)
hist(edx$userFreq, breaks = 100)
hist(edx$userAverageRating, breaks = 50)
hist(edx$movieAverageRating, breaks = 50)


ggplot(data = edxTrainSet) +
  geom_point(aes(x = Adventure, y = year))




#Delete columns


set.seed(755, sample.kind = 'Rounding')
test_index <- createDataPartition(y = edxTestSet$rating, times = 1, 
                                  p = 0.01, list = FALSE)
sampleSet <- edxTestSet[test_index,]

sampleSet <- sampleSet %>%
  mutate(row = row_number()) %>%
  separate_rows(genres, sep = '\\|') %>%
  pivot_wider(names_from = genres, values_from = genres, 
              values_fn = function(x) 1, values_fill = 0) %>%
  select(-row)


sampleSet <- sampleSet[-c(4:5,9:11)]

sampleSet <- sampleSet[-c(7:26)]


#####################
#Matrix Factorization
#####################

set.seed(755, sample.kind = 'Rounding')
sampletest_index <- createDataPartition(y = sampleSet2$rating, times = 1, 
                                  p = 0.01, list = FALSE)
sampletrain <- sampleSet2[-sampletest_index,]
sampletest <- sampleSet2[sampletest_index,]


recommender <- Reco()
recommender$sampletrain(sampletrain, opts = c(dim = 30, costp_12 = 0.1, costq_12 = 0.1,
                                       lrate = 0.1, niter = 100, nthread = 6, verbose = F))
sampletest$prediction <- recommender$predict(sampletest, out_memory())






##################################
#Run Parallel Matrix Factorization
##################################

#Run Matrix Factorization
library(recosystem)
set.seed(755, sample.kind = 'Rounding')
train_set = data_memory(edxTrainSet$userId, edxTrainSet$userAverageRating,
                        edxTrainSet$movieAverageRating, rating = edxTrainSet$rating, index1 = TRUE)
test_set = data_memory(edxTrainSet$userId, edxTrainSet$userAverageRating,
                       edxTrainSet$movieAverageRating, rating = NULL, index1 = TRUE)

r = Reco()
opts = r$train(train_set, opts = list(dim = 30, lrate = 0.1,
                                      costp_l2 = 0.1, costq_l2 = 0.1,
                                      nthread = 6, niter = 100, nthread = 6,
                                      verbose = F))
opts







#####
#PCA
#####

genresdf <- edxTrainSet[, 12:31]

set.seed(755, sample.kind = 'Rounding')
test_index <- createDataPartition(y = genresdf$Adventure, times = 1, 
                                  p = 0.2, list = FALSE)

pcatrain <- genresdf[-test_index,]
pcatest <- genresdf[test_index,]

pcatest <- pcatest %>%
  semi_join(pcatrain, by = "Adventure")

prin_comp <- prcomp(pcatrain, scale. = T)

names(prin_comp)

#outputs the mean of variables
prin_comp$center

#outputs the standard deviation of variables
prin_comp$scale

prin_comp$rotation

biplot(prin_comp, scale = 0)






####################################
#First Model: Movie and User Effects
####################################

#Setting Loss Function (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}
############
#Naive Model
############

mu_hat <- mean(edxTrainSet$rating)
RMSE(edxTestSet$rating, mu_hat)

#######################
#Modeling movie effects
#######################

mu <- mean(edxTrainSet$rating) 
movie_avgs <- edxTrainSet %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + edxTestSet %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
RMSE(predicted_ratings, edxTestSet$rating)

#####################
#Movie + User effects
#####################

user_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- edxTestSet %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
RMSE(predicted_ratings, edxTestSet$rating)

########################
#Penalized least squares
########################

#setting mu
lambda <- 0
mu <- mean(edxTrainSet$rating)
movie_reg_avgs <- edxTrainSet %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

#Predict ratings
predicted_ratings <- edxTestSet %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
RMSE(predicted_ratings, edxTestSet$rating)

#Set-Up Penalty Terms (Lambda)
lambdas <- seq(0, 10, 0.25)

mu <- mean(edxTrainSet$rating)
just_the_sum <- edxTrainSet %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- edxTrainSet %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edxTrainSet$rating))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]

######################################
#Movie + User effects + User Frequency
######################################

userfreq_avgs <- edxTrainSet %>% 
  left_join(user_avgs, by='userId') %>%
  group_by(userFreq) %>%
  summarize(b_uf = mean(rating - mu - b_u))

predicted_ratings <- edxTestSet %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(userfreq_avgs, by='userFreq') %>%
  mutate(pred = mu + b_i + b_u + b_uf) %>%
  pull(pred)
RMSE(predicted_ratings, edxTestSet$rating)






####################
#Ordinal Regression
####################


require(foreign)
require(ggplot2)
require(MASS)
require(Hmisc)
require(reshape2)

ordfit <-  polr(as.factor(rating) ~ userAverageRating + userFreq + movieAverageRating + movieFreq + year + 
                  as.factor(Adventure) + as.factor(Animation) + as.factor(Children) + 
                  as.factor(Comedy) + as.factor(Fantasy) + as.factor(Romance) + as.factor(Drama) +
                  as.factor(Action) + as.factor(Crime) + as.factor(Thriller) + as.factor(Horror) +
                  as.factor(Mystery) + as.factor(SciFi) + as.factor(IMAX) + as.factor(Documentary) + 
                  as.factor(War) + as.factor(Musical) + as.factor(FilmNoir) + as.factor(Western), data = edxTrainSet)
summary(ordfit) #adjusted r-square = .3262
confint(ordfit, level = .90)