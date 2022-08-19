#Separate the values of genres into columns(variable) per genre
movielens <- movielens %>%
  mutate(row = row_number()) %>%
  separate_rows(genres, sep = '\\|') %>%
  pivot_wider(names_from = genres, values_from = genres, 
              values_fn = function(x) 1, values_fill = 0) %>%
  select(-row)