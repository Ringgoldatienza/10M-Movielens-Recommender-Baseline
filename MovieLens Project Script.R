################################################################################
#title: "MovieLens Recommendation System Project"
#subtitle: "HarvardX - PH125.9x: Data Science: Capstone"
#author: "Ringgold P. Atienza"
################################################################################

################################################################################
# Download Movielens Dataset - code provided by HarvardX: PH125.9x
################################################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
if(!require(forcats)) install.packages("forcats", repos = "http://cran.us.r-project.org")

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

################################################################################
#Mutate dataset to flesh out relevant variables for the model
################################################################################

#Data manipulation: Extract the year of the release of the movie
movielens <- mutate(movielens, title = str_trim(title)) %>%
  extract(title, c("title_temp", "movieYear"), regex = "^(.*) \\(([0-9 \\-]*)\\)$", 
          remove = F) %>%
  mutate(movieYear = if_else(str_length(movieYear) > 4, 
                             as.integer(str_split(movieYear, 
                                                  "-", simplify = T)[1]), 
                             as.integer(movieYear))) %>%
  mutate(title = if_else(is.na(title_temp), title, title_temp)) %>%
  select(-title_temp)

#Data manipulation: Mutate timestamp into dates and years
movielens <- mutate(movielens, reviewDate = round_date(as_datetime(timestamp), unit = "week"))
movielens <- mutate(movielens, reviewYear = year(as_datetime(reviewDate)))

#Data manipulation: Create age of the movie during review variable
movielens <- mutate(movielens, movieAge = reviewYear - movieYear)

#Data manipulation: add no. of times users rate movies
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
movielens <- ddply(movielens, .(userId), transform, user = count(userId))
movielens <- subset(movielens, select = -c(user.x))
setnames(movielens, "user.freq", "userFreq")

#Data manipulation: add mo.of times movies are rated
movielens <- ddply(movielens, .(movieId), transform, movie = count(movieId))
movielens <- subset(movielens, select = -c(movie.x))
setnames(movielens, "movie.freq", "movieFreq")

detach(package:plyr) #unload plyr package as it can cause compatibility problems in other packages.

################################################################################
#Partition Movielens to edx(for training and testing) and validation set
################################################################################

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId") %>%
  semi_join(edx, by = "movieFreq") %>%
  semi_join(edx, by = "movieAge") %>%
  semi_join(edx, by = "movieYear") %>%
  semi_join(edx, by = "userFreq") %>%
  semi_join(edx, by = "genres")
  
# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################################################################
#Summaries and visual inspection of the variables
################################################################################

#Summarize data set (first 10 rows)
head(edx, 10)
summary(edx)

#Summarize number of users, movies, and ratings
edx %>% summarize(n_users = n_distinct(userId),
                  n_movies = n_distinct(movieId),
                  n_ratings = nrow(edx))

#Plot Figure 1. Actual rating distribution
ggplot(edx, aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "black") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  labs(x = "Rating", y = "Count",
       subtitle = "n = 90,000,055 ratings",
       caption = "*based on edx dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12))

#Plot Figure 2. Distribution of average ratings per movie
edx %>% group_by(movieId) %>%
  summarise(ave_rating = sum(rating)/n()) %>%
  ggplot(aes(ave_rating)) +
  geom_histogram(binwidth = .10, color = "black") +
  labs(x = "Average rating", y = "Number of movies",
       subtitle = "n = 10,677 movies",
       caption = "*based on edx dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
    plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12))

#Plot Figure 3. Distribution of average ratings by user
edx %>% group_by(userId) %>%
  summarise(ave_rating = sum(rating)/n()) %>%
  ggplot(aes(ave_rating)) +
  geom_histogram(binwidth = .10, color = "black") +
  labs(x = "Average rating", y = "Number of users",
       subtitle = "n = 69,878 users",
       caption = "*based on edx dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12))

#Plot Figure 4. Distribution of ratings by year
ggplot(edx, aes(movieYear)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_y_continuous(breaks = seq(0, 8000000, 100000), labels = seq(0, 80000, 1000)) +
  labs(x = "Movie Year", y = "Count ('000s)", caption = "*based on edx dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12))

#Plot Figure 5. Distribution of rating by movie age
ggplot(edx, aes(movieAge)) +
  geom_histogram(binwidth = 1, color = "black") +
  scale_y_continuous(breaks = seq(0, 1100000, 100000), labels = seq(0, 1100, 100)) +
  labs(x = "Movie Age", y = "Count (,000s)", caption = "*based on edx dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12))

#Arrange plot from highest value to lowest value
#source: https://blog.albertkuo.me/post/2022-01-04-reordering-geom-col-and-geom-bar-by-count-or-value/

#Plot Figure 6. Distribution of rating by genre
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  ggplot(aes(x = fct_infreq(genres))) +
  geom_bar() +
  scale_y_continuous(breaks = seq(0, 4000000, 500000)) +
  labs(x = "Genre", y = "Count", caption = "*based on edx dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
      plot.caption = element_text(size = 12, face = "italic"), 
      axis.title = element_text(size = 12),
      axis.text.x = element_text(angle = 60, hjust =1))

################################################################################
#Partition the edx into train and test Set
################################################################################

#Assign edx data into train (80%) and test (20%) sets.
set.seed(2022, sample.kind = 'Rounding')
test_index <- createDataPartition(y = edx$rating, times = 1, 
                                  p = 0.2, list = FALSE)
edxTrainSet <- edx[-test_index,]
temp <- edx[test_index,]

#To overlap the movie and user Ids using semi_join
edxTestSet <- temp %>%
  semi_join(edxTrainSet, by = "movieId") %>%
  semi_join(edxTrainSet, by = "userId") %>%
  semi_join(edxTrainSet, by = "movieFreq") %>%
  semi_join(edxTrainSet, by = "movieAge") %>%
  semi_join(edxTrainSet, by = "movieYear") %>%
  semi_join(edxTrainSet, by = "userFreq") %>%
  semi_join(edxTrainSet, by = "genres")
  
rm(test_index, temp)

################################################################################
#Stepwise modelling (baseline)
################################################################################

#Setting Loss Function (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
  }

#Calculate overall average as baseline rating
mu <- mean(edxTrainSet$rating)

#Test prediction
rmse_baseline <- RMSE(edxTestSet$rating, mu)

rmse_baseline_step  <- data.frame(Variable = "Baseline (mu)", 
                           RMSE = rmse_baseline, 
                           Difference = rmse_baseline - rmse_baseline)

################################################################################
#Predict (mu + bi)
movie_avgs <- edxTrainSet %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + edxTestSet %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_rmse <- RMSE(predicted_ratings, edxTestSet$rating) 

movie_rmse <- data.frame(Variable = "Baseline + Movie Effect (mu + b_i)", 
                 RMSE = movie_rmse,
                 Difference = rmse_baseline - movie_rmse)

rmse_baseline_step  <- rbind(rmse_baseline_step , movie_rmse)

rm(movie_rmse)

################################################################################
#Predict (mu + bu)
user_avgs <- edxTrainSet %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu))

predicted_ratings <- mu + edxTestSet %>% 
  left_join(user_avgs, by='userId') %>%
  pull(b_u)

user_rmse <- RMSE(predicted_ratings, edxTestSet$rating)

user_rmse <- data.frame(Variable = "Baseline + User Effect (mu + b_u)", 
                           RMSE = user_rmse,
                           Difference = rmse_baseline - user_rmse)

rmse_baseline_step  <- rbind(rmse_baseline_step , user_rmse)

rm(user_rmse)

################################################################################
#Predict (mu + b_ma)
movieage_avgs <- edxTrainSet %>% 
  group_by(movieAge) %>% 
  summarize(b_ma = mean(rating - mu))

predicted_ratings <- mu + edxTestSet %>% 
  left_join(movieage_avgs, by='movieAge') %>%
  pull(b_ma)

movieage_rmse <- RMSE(predicted_ratings, edxTestSet$rating)

movieage_rmse <- data.frame(Variable = "Baseline + Movie Age Effect (mu + b_ma)", 
                           RMSE = movieage_rmse ,
                           Difference = rmse_baseline - movieage_rmse )

rmse_baseline_step  <- rbind(rmse_baseline_step , movieage_rmse)
rm(movieage_rmse)

################################################################################
#Predict (mu + b_my)
movieyear_avgs <- edxTrainSet %>% 
  group_by(movieYear) %>%
  summarize(b_my = mean(rating - mu))

predicted_ratings <- mu + edxTestSet %>% 
  left_join(movieyear_avgs, by='movieYear') %>%
  pull(b_my)

movieyear_rmse <- RMSE(predicted_ratings, edxTestSet$rating)

movieyear_rmse <- data.frame(Variable = "Baseline + Movie Year Effect (mu + b_my)", 
                               RMSE = movieyear_rmse ,
                               Difference = rmse_baseline - movieyear_rmse )

rmse_baseline_step  <- rbind(rmse_baseline_step , movieyear_rmse)
rm(movieyear_rmse)

################################################################################
#Predict (mu + b_g)
genres_avgs <- edxTrainSet %>% 
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu))

predicted_ratings <- mu + edxTestSet %>% 
  left_join(genres_avgs, by='genres') %>%
  pull(b_g)

genres_rmse <- RMSE(predicted_ratings, edxTestSet$rating)

genres_rmse <- data.frame(Variable = "Baseline + Genres Effect (mu + b_g)", 
                                RMSE = genres_rmse ,
                                Difference = rmse_baseline - genres_rmse )

rmse_baseline_step  <- rbind(rmse_baseline_step , genres_rmse)
rm(genres_rmse)

################################################################################
#Predict (mu + b_uf)
userfreq_avgs <- edxTrainSet %>% 
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu))

predicted_ratings <- mu + edxTestSet %>% 
  left_join(userfreq_avgs, by='userFreq') %>%
  pull(b_uf)

userfreq_rmse <- RMSE(predicted_ratings, edxTestSet$rating)

userfreq_rmse <- data.frame(Variable = "Baseline + User Frequency Effect (mu + b_uf)", 
                           RMSE = userfreq_rmse,
                           Difference = rmse_baseline - userfreq_rmse)

rmse_baseline_step  <- rbind(rmse_baseline_step , userfreq_rmse)
rm(userfreq_rmse)

################################################################################
#Predict (mu + b_mf)
moviefreq_avgs <- edxTrainSet %>% 
  group_by(movieFreq) %>% 
  summarize(b_mf = mean(rating - mu))

predicted_ratings <- mu + edxTestSet %>% 
  left_join(moviefreq_avgs, by='movieFreq') %>%
  pull(b_mf)

moviefreq_rmse <- RMSE(predicted_ratings, edxTestSet$rating)

moviefreq_rmse <- data.frame(Variable = "Baseline + Movie Frequency Effect (mu + b_mf)", 
                               RMSE = moviefreq_rmse,
                               Difference = rmse_baseline - moviefreq_rmse)

rmse_baseline_step  <- rbind(rmse_baseline_step , moviefreq_rmse)
rm(moviefreq_rmse)
rmse_baseline_step 

################################################################################
#Plot Figure 7. Stepwise (baseline) RMSE values
rmse_baseline_step <- rmse_baseline_step [order(-rmse_baseline_step $RMSE),]

ggplot(rmse_baseline_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Variable)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12),
        axis.text.x = element_text(angle = 60, hjust =1))

################################################################################
#Stepwise modelling (baseline + movie)
################################################################################

#Predict (mu + b_i)
movie_avgs <- edxTrainSet %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_ratings <- mu + edxTestSet %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

rmse_movie_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_movie_step <- data.frame(Stepwise = "Baseline + Movie", 
                              RMSE = rmse_movie_step_temp ,
                              Difference = 0 )

################################################################################
#Predict (mu + b_i + b_mf)
moviefreq_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(movieFreq) %>% 
  summarize(b_mf = mean(rating - mu - b_i))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  mutate(pred = mu + b_i + b_mf) %>%
  pull(pred)

rmse_movie_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_movie_step_temp <- data.frame(Stepwise = "+ Movie Frequency", 
                             RMSE = rmse_movie_step_temp,
                             Difference = rmse_movie_step[1,2] - rmse_movie_step_temp)

rmse_movie_step <- rbind(rmse_movie_step, rmse_movie_step_temp)

################################################################################
#Predict (mu + b_i + b_u)
user_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_movie_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_movie_step_temp <- data.frame(Stepwise = "+ Movie + User", 
                                   RMSE = rmse_movie_step_temp,
                                   Difference = rmse_movie_step[1,2] - rmse_movie_step_temp)

rmse_movie_step <- rbind(rmse_movie_step, rmse_movie_step_temp)

################################################################################
#Predict (mu + b_i + b_g)
genres_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_g) %>%
  pull(pred)

rmse_movie_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_movie_step_temp <- data.frame(Stepwise = "+ Genres", 
                                   RMSE = rmse_movie_step_temp,
                                   Difference = rmse_movie_step[1,2] - rmse_movie_step_temp)

rmse_movie_step <- rbind(rmse_movie_step, rmse_movie_step_temp)

################################################################################
#Predict (mu + b_i + b_uf)

userfreq_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu - b_i))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  mutate(pred = mu + b_i + b_uf) %>%
  pull(pred)

rmse_movie_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_movie_step_temp <- data.frame(Stepwise = "+ User Frequency", 
                                   RMSE = rmse_movie_step_temp,
                                   Difference = rmse_movie_step[1,2] - rmse_movie_step_temp)

rmse_movie_step <- rbind(rmse_movie_step, rmse_movie_step_temp)

################################################################################
#Predict (mu + b_i + b_my)
movieyear_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(movieYear) %>% 
  summarize(b_my = mean(rating - mu - b_i))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  mutate(pred = mu + b_i + b_my) %>%
  pull(pred)

rmse_movie_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_movie_step_temp <- data.frame(Stepwise = "+ Movie Year", 
                                   RMSE = rmse_movie_step_temp,
                                   Difference = rmse_movie_step[1,2] - rmse_movie_step_temp)

rmse_movie_step <- rbind(rmse_movie_step, rmse_movie_step_temp)

################################################################################
#Predict (mu + b_i + b_ma)
movieage_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(movieAge) %>% 
  summarize(b_ma = mean(rating - mu - b_i))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  mutate(pred = mu + b_i + b_ma) %>%
  pull(pred)

rmse_movie_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_movie_step_temp <- data.frame(Stepwise = "+ Movie Age", 
                                   RMSE = rmse_movie_step_temp,
                                   Difference = rmse_movie_step[1,2] - rmse_movie_step_temp)

rmse_movie_step <- rbind(rmse_movie_step, rmse_movie_step_temp)
rmse_movie_step

################################################################################
#Plot Figure 8. Stepwise (baseline + movie) RMSE values
rmse_movie_step <- rmse_movie_step [order(-rmse_movie_step $RMSE),]

ggplot(rmse_movie_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12),
        axis.text.x = element_text(angle = 60, hjust =1))

################################################################################
#Stepwise modelling (baseline + movie + user)
################################################################################

#Predict (mu + b_i + b_u)
user_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_user_step <- data.frame(Stepwise = "Baseline + Movie + User", 
                              RMSE = rmse_user_step_temp,
                              Difference = 0)

################################################################################
#Predict (mu + b_i + b_u + b_uf)
userfreq_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu - b_i - b_u))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  mutate(pred = mu + b_i + b_u + b_uf) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_user_step_temp <- data.frame(Stepwise = "+ User Frequency", 
                             RMSE = rmse_user_step_temp,
                             Difference = rmse_user_step[1,2] - rmse_user_step_temp)

rmse_user_step <- rbind(rmse_user_step, rmse_user_step_temp)

################################################################################
#Predict (mu + b_i + b_u + b_ma)
movieage_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(movieAge) %>% 
  summarize(b_ma = mean(rating - mu - b_i - b_u))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  mutate(pred = mu + b_i + b_u + b_ma) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_user_step_temp <- data.frame(Stepwise = "+ Movie Age", 
                                  RMSE = rmse_user_step_temp,
                                  Difference = rmse_user_step[1,2] - rmse_user_step_temp)

rmse_user_step <- rbind(rmse_user_step, rmse_user_step_temp)

################################################################################
#Predict (mu + b_i + b_u + b_my)
movieyear_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(movieYear) %>% 
  summarize(b_my = mean(rating - mu - b_i - b_u))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  mutate(pred = mu + b_i + b_u + b_my) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_user_step_temp <- data.frame(Stepwise = "+ Movie Year", 
                                  RMSE = rmse_user_step_temp,
                                  Difference = rmse_user_step[1,2] - rmse_user_step_temp)

rmse_user_step <- rbind(rmse_user_step, rmse_user_step_temp)

################################################################################
#Predict (mu + b_i + b_u + b_g)
genres_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_g) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_user_step_temp <- data.frame(Stepwise = "+ Genres", 
                                  RMSE = rmse_user_step_temp,
                                  Difference = rmse_user_step[1,2] - rmse_user_step_temp)

rmse_user_step <- rbind(rmse_user_step, rmse_user_step_temp)

################################################################################
#Predict (mu + b_i + b_u + b_mf)
moviefreq_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(movieFreq) %>% 
  summarize(b_mf = mean(rating - mu - b_i - b_u))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf) %>%
  pull(pred)

rmse_user_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_user_step_temp <- data.frame(Stepwise = "+ Movie Frequency", 
                                  RMSE = rmse_user_step_temp,
                                  Difference = rmse_user_step[1,2] - rmse_user_step_temp)

rmse_user_step <- rbind(rmse_user_step, rmse_user_step_temp)
rmse_user_step

################################################################################
#Plot Figure 8. Stepwise (baseline + movie + user) RMSE values
rmse_user_step <- rmse_user_step [order(-rmse_user_step $RMSE),]

ggplot(rmse_user_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12),
        axis.text.x = element_text(angle = 60, hjust =1))

################################################################################
#Stepwise modelling (baseline + movie + user + movie frequency)
################################################################################

#Predict (mu + b_i + b_u + b_mf)
moviefreq_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  group_by(movieFreq) %>% 
  summarize(b_mf = mean(rating - mu - b_i - b_u))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf) %>%
  pull(pred)

rmse_mf_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_mf_step <- data.frame(Stepwise = "Baseline + Movie + User + Movie Frequency", 
                             RMSE = rmse_mf_step_temp,
                             Difference = 0)

################################################################################
#Predict (mu + b_i + b_u + b_mf + b_ma)
movieage_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  group_by(movieAge) %>% 
  summarize(b_ma = mean(rating - mu - b_i - b_u - b_mf))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma) %>%
  pull(pred)

rmse_mf_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_mf_step_temp <- data.frame(Stepwise = "+ Movie Age", 
                                  RMSE = rmse_mf_step_temp,
                                  Difference = rmse_mf_step[1,2] - rmse_mf_step_temp)

rmse_mf_step <- rbind(rmse_mf_step, rmse_mf_step_temp)

################################################################################
#Predict (mu + b_i + b_u + b_mf + b_g)
genres_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_mf))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_g) %>%
  pull(pred)

rmse_mf_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_mf_step_temp <- data.frame(Stepwise = "+ Genres", 
                                RMSE = rmse_mf_step_temp,
                                Difference = rmse_mf_step[1,2] - rmse_mf_step_temp)

rmse_mf_step <- rbind(rmse_mf_step, rmse_mf_step_temp)

################################################################################
#Predict (mu + b_i + b_u + b_mf + b_my)
movieyear_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  group_by(movieYear) %>% 
  summarize(b_my = mean(rating - mu - b_i - b_u - b_mf))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_my) %>%
  pull(pred)

rmse_mf_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_mf_step_temp <- data.frame(Stepwise = "+ Movie Year", 
                                RMSE = rmse_mf_step_temp,
                                Difference = rmse_mf_step[1,2] - rmse_mf_step_temp)

rmse_mf_step <- rbind(rmse_mf_step, rmse_mf_step_temp)

################################################################################
#Predict (mu + b_i + b_u + b_mf + b_uf)
userfreq_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu - b_i - b_u - b_mf))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_uf) %>%
  pull(pred)

rmse_mf_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_mf_step_temp <- data.frame(Stepwise = "+ User Frequency", 
                                RMSE = rmse_mf_step_temp,
                                Difference = rmse_mf_step[1,2] - rmse_mf_step_temp)

rmse_mf_step <- rbind(rmse_mf_step, rmse_mf_step_temp)
rmse_mf_step

################################################################################
#Plot Figure 8. Stepwise (baseline + movie + user + movie frequency) RMSE values
rmse_mf_step <- rmse_mf_step [order(-rmse_mf_step $RMSE),]

ggplot(rmse_mf_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12),
        axis.text.x = element_text(angle = 60, hjust =1))

################################################################################
#Stepwise modelling (baseline + movie + user + movie frequency + movie age)
################################################################################

#Predict (mu + b_i + b_u + b_mf + b_ma)
movieage_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  group_by(movieAge) %>% 
  summarize(b_ma = mean(rating - mu - b_i - b_u - b_mf))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma) %>%
  pull(pred)

rmse_ma_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_ma_step <- data.frame(Stepwise = "Baseline + ... + Movie Age", 
                           RMSE = rmse_ma_step_temp,
                           Difference = 0)

################################################################################
#Predict (mu + b_i + b_u + b_mf + b_ma + b_uf)
userfreq_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu - b_i - b_u - b_mf - b_ma))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_uf) %>%
  pull(pred)

rmse_ma_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_ma_step_temp <- data.frame(Stepwise = "+ User Frequency", 
                                RMSE = rmse_ma_step_temp,
                                Difference = rmse_ma_step[1,2] - rmse_ma_step_temp)

rmse_ma_step <- rbind(rmse_ma_step, rmse_ma_step_temp)

################################################################################
#Predict (mu + b_i + b_u + b_mf + b_ma + b_g)
genres_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_mf - b_ma))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_g) %>%
  pull(pred)

rmse_ma_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_ma_step_temp <- data.frame(Stepwise = "+ Genres", 
                                RMSE = rmse_ma_step_temp,
                                Difference = rmse_ma_step[1,2] - rmse_ma_step_temp)

rmse_ma_step <- rbind(rmse_ma_step, rmse_ma_step_temp)

################################################################################
#Predict (mu + b_i + b_u + b_mf + b_ma + b_my)
movieyear_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  group_by(movieYear) %>% 
  summarize(b_my = mean(rating - mu - b_i - b_u - b_mf - b_ma))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_my) %>%
  pull(pred)

rmse_ma_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_ma_step_temp <- data.frame(Stepwise = "+ Movie Year", 
                                RMSE = rmse_ma_step_temp,
                                Difference = rmse_ma_step[1,2] - rmse_ma_step_temp)

rmse_ma_step <- rbind(rmse_ma_step, rmse_ma_step_temp)
rmse_ma_step

################################################################################
#Plot Figure 9. Stepwise (baseline + movie + user + movie frequency) RMSE values
rmse_ma_step <- rmse_ma_step [order(-rmse_ma_step $RMSE),]

ggplot(rmse_ma_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12),
        axis.text.x = element_text(angle = 60, hjust =1))

################################################################################
#Stepwise modelling (baseline + movie + user + movie frequency + movie age + 
#     movie year)
################################################################################

#Predict (mu + b_i + b_u + b_mf + b_ma + b_my)
movieyear_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  group_by(movieYear) %>% 
  summarize(b_my = mean(rating - mu - b_i - b_u - b_mf - b_ma))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_my) %>%
  pull(pred)

rmse_my_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_my_step <- data.frame(Stepwise = "Baseline + ... + Movie Year", 
                           RMSE = rmse_my_step_temp,
                           Difference = 0)

################################################################################
#Predict (mu + b_i + b_u + b_mf + b_ma + b_my + b_uf)
userfreq_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu - b_i - b_u - b_mf - b_ma - b_my))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_my + b_uf) %>%
  pull(pred)

rmse_my_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_my_step_temp <- data.frame(Stepwise = "+ User Frequency", 
                                RMSE = rmse_my_step_temp,
                                Difference = rmse_my_step[1,2] - rmse_my_step_temp)

rmse_my_step <- rbind(rmse_my_step, rmse_my_step_temp)

################################################################################
#Predict (mu + b_i + b_u + b_mf + b_ma + b_my + b_g)
genres_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_mf - b_ma - b_my))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_my + b_g) %>%
  pull(pred)

rmse_my_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_my_step_temp <- data.frame(Stepwise = "+ Genres", 
                                RMSE = rmse_my_step_temp,
                                Difference = rmse_my_step[1,2] - rmse_my_step_temp)

rmse_my_step <- rbind(rmse_my_step, rmse_my_step_temp)

################################################################################
#Plot Figure 10. Stepwise (baseline + movie + user + movie frequency + movie year) RMSE values
rmse_my_step <- rmse_my_step [order(-rmse_my_step $RMSE),]

ggplot(rmse_my_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12),
        axis.text.x = element_text(angle = 60, hjust =1))

################################################################################
#Stepwise modelling (baseline + movie + user + movie frequency + movie age + 
#     movie year + user frequency)
################################################################################

#Predict (mu + b_i + b_u + b_mf + b_ma + b_my + b_uf)
userfreq_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  group_by(userFreq) %>% 
  summarize(b_uf = mean(rating - mu - b_i - b_u - b_mf - b_ma - b_my))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_my + b_uf) %>%
  pull(pred)

rmse_uf_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_uf_step <- data.frame(Stepwise = "Baseline + ... + User Frequency", 
                           RMSE = rmse_uf_step_temp,
                           Difference = 0)

################################################################################
#Predict (mu + b_i + b_u + b_mf + b_ma + b_my + b_uf + b_g)
genres_avgs <- edxTrainSet %>% 
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  group_by(genres) %>% 
  summarize(b_g = mean(rating - mu - b_i - b_u - b_mf - b_ma - b_my - b_uf))

predicted_ratings <- edxTestSet %>%
  left_join(movie_avgs, by = "movieId") %>%
  left_join(user_avgs, by = "userId") %>%
  left_join(moviefreq_avgs, by = "movieFreq") %>%
  left_join(movieage_avgs, by = "movieAge") %>%
  left_join(movieyear_avgs, by = "movieYear") %>%
  left_join(userfreq_avgs, by = "userFreq") %>%
  left_join(genres_avgs, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_my + b_uf + b_g) %>%
  pull(pred)

rmse_fullmodel <- RMSE(predicted_ratings, edxTestSet$rating)
rmse_uf_step_temp <- RMSE(predicted_ratings, edxTestSet$rating) 
rmse_uf_step_temp <- data.frame(Stepwise = "+ Genres", 
                                RMSE = rmse_uf_step_temp,
                                Difference = rmse_uf_step[1,2] - rmse_uf_step_temp)

rmse_uf_step <- rbind(rmse_uf_step, rmse_uf_step_temp)
fullmodel_predicted_ratings <- predicted_ratings

################################################################################
#Plot Figure 10. Stepwise (baseline + movie + user + movie frequency + 
#     movie year + user frequency) RMSE values
rmse_uf_step <- rmse_uf_step [order(-rmse_uf_step $RMSE),]

ggplot(rmse_uf_step , aes(x = RMSE, y = forcats::fct_inorder(as.factor(Stepwise)))) +
  geom_point(stat = "identity") +
  labs(x = "RMSE Values", y = "", caption = "*based on edx training dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 12, face = "italic"), 
        axis.title = element_text(size = 12),
        axis.text.x = element_text(angle = 60, hjust =1))

################################################################################
#Apply regularization using penalized least squares
################################################################################

#Set cross validation for the tuning parameter (lambda)
lambdas <- seq(0, 3, .1)

#Regularize the final model: y = mu + b_i + b_u + b_mf + b_ma + b_my + b_uf + b_g
rmses <- sapply(lambdas, function(l){
  b_i <- edxTrainSet %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/n() + l)
  b_u <- edxTrainSet %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/n() + l)
  b_mf <- edxTrainSet %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(movieFreq) %>%
    summarise(b_mf = sum(rating - b_i - b_u - mu)/n() + l)
  b_ma <- edxTrainSet %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_mf, by = "movieFreq") %>%
    group_by(movieAge) %>%
    summarise(b_ma = sum(rating - b_i - b_u - b_mf - mu)/n() + l)
  b_my <- edxTrainSet %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_mf, by = "movieFreq") %>%
    left_join(b_ma, by = "movieAge") %>%
    group_by(movieYear) %>%
    summarise(b_my = sum(rating - b_i - b_u - b_mf - b_ma - mu)/n() + l)
  b_uf <- edxTrainSet %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_mf, by = "movieFreq") %>%
    left_join(b_ma, by = "movieAge") %>%
    left_join(b_my, by = "movieYear") %>%
    group_by(userFreq) %>%
    summarise(b_uf = sum(rating - b_i - b_u - b_mf - b_ma - b_my - mu)/n() + l)
  b_g <- edxTrainSet %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_mf, by = "movieFreq") %>%
    left_join(b_ma, by = "movieAge") %>%
    left_join(b_my, by = "movieYear") %>%
    left_join(b_uf, by = "userFreq") %>%
    group_by(genres) %>%
    summarise(b_g = sum(rating - b_i - b_u - b_mf - b_ma - b_my - b_uf - mu)/n() + l)
  predicted_ratings <- edxTestSet %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_mf, by = "movieFreq") %>%
    left_join(b_ma, by = "movieAge") %>%
    left_join(b_my, by = "movieYear") %>%
    left_join(b_uf, by = "userFreq") %>%
    left_join(b_g, by = "genres") %>%
    mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_my + b_uf + b_g) %>%
    pull(pred)
    return(RMSE(predicted_ratings, edxTestSet$rating))
})

#Plot lambda and RMSE
penalty_terms <- data.frame(lambdas, rmses)
ggplot(penalty_terms, aes(x = lambdas, y = rmses)) +
  geom_point() +
  labs(x = "Lambda", y = "RMSE", caption = "*based on final training dataset") +
  theme(plot.title = element_text(size = rel(1.5)),
        plot.caption = element_text(size = 8, face = "italic"), 
        axis.title = element_text(size = 12)) 
#lambda is set to zero, therefore no penalty terms is set to the model

rmse_regularized <- min(rmses)

################################################################################
#Final hold-out test of the complete model
################################################################################
#Create predictions for the Final Model against Validation dataset
b_i <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/n())
b_u <- edx %>%
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - b_i - mu)/n())
b_mf <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(movieFreq) %>%
  summarise(b_mf = sum(rating - b_i - b_u - mu)/n())
b_ma <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_mf, by = "movieFreq") %>%
  group_by(movieAge) %>%
  summarise(b_ma = sum(rating - b_i - b_u - b_mf - mu)/n())
b_my <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_mf, by = "movieFreq") %>%
  left_join(b_ma, by = "movieAge") %>%
  group_by(movieYear) %>%
  summarise(b_my = sum(rating - b_i - b_u - b_mf - b_ma - mu)/n())
b_uf <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_mf, by = "movieFreq") %>%
  left_join(b_ma, by = "movieAge") %>%
  left_join(b_my, by = "movieYear") %>%
  group_by(userFreq) %>%
  summarise(b_uf = sum(rating - b_i - b_u - b_mf - b_ma - b_my - mu)/n())
b_g <- edx %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_mf, by = "movieFreq") %>%
  left_join(b_ma, by = "movieAge") %>%
  left_join(b_my, by = "movieYear") %>%
  left_join(b_uf, by = "userFreq") %>%
  group_by(genres) %>%
  summarise(b_g = sum(rating - b_i - b_u - b_mf - b_ma - b_my - b_uf - mu)/n())
predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_mf, by = "movieFreq") %>%
  left_join(b_ma, by = "movieAge") %>%
  left_join(b_my, by = "movieYear") %>%
  left_join(b_uf, by = "userFreq") %>%
  left_join(b_g, by = "genres") %>%
  mutate(pred = mu + b_i + b_u + b_mf + b_ma + b_my + b_uf + b_g) %>%
  pull(pred)

#Test Final Model and get the value of RMSE
rmse_final <- RMSE(predicted_ratings, validation$rating) 
rmse_final
