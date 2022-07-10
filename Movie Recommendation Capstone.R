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

