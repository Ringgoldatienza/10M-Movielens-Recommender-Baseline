##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(lubridate)
library(tidyr)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")


# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

##################
#Model Development
##################

###########################
#Create variables/Features
##########################


#No of times users rate movies
edx <- ddply(edx, .(userId), transform, user = count(userId))
edx <- subset(edx, select = -c(user.x))
colnames(edx)[7] <- "userFreq"

#Average rating of movies per user
edx <- ddply(edx, .(userId), transform, userAverageRating = mean(rating))

#Average rating per movie
edx <- ddply(edx, .(movieId), transform, movieAverageRating = mean(rating))

#Mutate timestamp into dates
edx <- mutate(edx, date = (as_datetime(timestamp)),
              year = year(as_datetime(timestamp)),
              month = month(as_datetime(timestamp)))

#Separate the values of genres into columns(variable) per genre
edx <- edx %>%
  mutate(row = row_number()) %>%
  separate_rows(genres, sep = '\\|') %>%
  pivot_wider(names_from = genres, values_from = genres, 
              values_fn = function(x) 1, values_fill = 0) %>%
  select(-row)

setnames(edx, "(no genres listed)", "No-Genre")

########################
#Train Set and Test Set
########################

#Assign edx data into train (80%) and test (20%) sets.
set.seed(755, sample.kind = 'Rounding')
test_index <- createDataPartition(y = edx$rating, times = 1, 
                                  p = 0.2, list = FALSE)

edxTrainSet <- edx[-test_index,]
edxTestSet <- edx[test_index,]

#To overlap the movie and user Ids using semi_join
edxTestSet <- edxTestSet %>%
  semi_join(edxTrainSet, by = "movieId") %>%
  semi_join(edxTrainSet, by = "userId")

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
#Regression Model 1
####################

fit <- lm(rating ~ userFreq + userAverageRating + movieAverageRating + year, data = edxTrainSet)
summary(fit)

predicted_ratings <- predict(fit, edxTestSet)
RMSE(predicted_ratings, edxTestSet$rating)

####################
#Regression Model 2
####################

fit <- lm(rating ~ userFreq + userAverageRating + movieAverageRating + year + as.factor(Adventure) + as.factor(Crime) + as.factor(Romance) + as.factor(Comedy), data = edxTrainSet)
summary(fit)

predicted_ratings <- predict(fit, edxTestSet)
RMSE(predicted_ratings, edxTestSet$rating)


library(car)
vif(fit)