#Separate the values of genres into columns(variable) per genre
movielens <- movielens %>%
  mutate(row = row_number()) %>%
  separate_rows(genres, sep = '\\|') %>%
  pivot_wider(names_from = genres, values_from = genres, 
              values_fn = function(x) 1, values_fill = 0) %>%
  select(-row)
              
              
##Delete non-UTF-8 encodings              
#The Movielens 1M data has some invalid utf8 encoding which can cause problems
count_invalidutf8 <- validUTF8(movielens$title)
nrow(movielens)-sum(count_invalidutf8, na.rm = TRUE) #There are 4199 titles with non UTF8 encoding

#Replace any non-UTF8 to ''
movielens$title <- iconv(movielens$title, "UTF-8", "UTF-8",sub='')
