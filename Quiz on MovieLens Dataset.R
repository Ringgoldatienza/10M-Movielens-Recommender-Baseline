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


#########################
#Quiz: MovieLens Dataset
#########################

#How many rows and columns are there in the edx dataset?
nrow(edx)
ncol(edx)

#How many zeroes were given as ratings in the edx dataset?
sum(with(edx, rating == 0))

#How many threes were given as ratings in the edx dataset?
sum(with(edx, rating == 3))

#How many different movies are in the edx dataset?
n_distinct(edx$movieId)

#How many different users are in the edx dataset?
n_distinct(edx$userId)

#How many movie ratings are in each of the following genres in the edx dataset?
sum(str_count(edx$genres, "Drama"))
sum(str_count(edx$genres, "Comedy"))
sum(str_count(edx$genres, "Thriller"))
sum(str_count(edx$genres, "Romance"))

#Which movie has the greatest number of ratings?
edx %>% group_by(movieId,title) %>% 
  summarise(n=n()) %>% 
  arrange(-n)

#What are the five most given ratings in order from most to least?
edx %>% count(rating, sort = TRUE)

#True or False: In general, half star ratings are less common than whole star ratings (e.g., there are fewer ratings of 3.5 than there are ratings of 3 or 4, etc.).
sum(edx$rating == 1.0, 2.0, 3.0, 4.0, 5.0) > sum(edx$rating == 1.5, 2.5, 3.5, 4.5)

