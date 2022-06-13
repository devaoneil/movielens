###########################################################
# Prediction Algorithm for Movie Ratings
# Implements predictions based on user bias, movie, 
# and user preferences for genre
###########################################################

##########################################################
# Step 1: Create edx set, validation set (final hold-out test set)
##########################################################
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

# Since I am using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
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
###########################################################
# Step 2: Explore edx data to identify promising predictors
###########################################################
library(stringr)
dim(edx) #How many rows and columns are there in the edx dataset?
count(edx,rating) #how many of each rating? 
nrow(distinct(edx,movieId)) #how many distinct movies
nrow(distinct(edx,userId)) #how many distinct users
nrow(distinct(validation,movieId)) #how many distinct movies
nrow(distinct(validation,userId)) #how many distinct users
dim(validation)
# The edx dataset has 9000055 observations, with 10,677
# distinct movies and 69,878 users. 

names(edx)
edx  %>% 
  group_by(userId) %>%  #explore user influence
  filter(n()>=100) %>% #select users with at least 100 ratings
  summarize(avg_by_user = mean(rating)) %>% #calculate average rating
  ggplot(aes(avg_by_user)) + 
  geom_histogram(bins = 30, color = "black")
# From the histogram, we see a lot of spread among users; 
# some are biased high and others low. 
# What about individual movies - do some average higher than others? To find out, we do a similar histogram for movie ratings: 
  
edx  %>% 
  group_by(movieId) %>%  #explore movie influence
  filter(n()>=100) %>% #select movies with at least 100 ratings
  summarize(avg_by_movie = mean(rating)) %>% #calculate average rating
  ggplot(aes(avg_by_movie)) + 
  geom_histogram(bins = 30, color = "black")

# From the histogram, we see a lot of spread among movies; many are biased high but others are very low. The movie bias will be calculated to help predict the ratings.
# a category will be defined as whatever combination appears in 
# genre (eg, "Adventure|Romance") 

categories<- edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating), se = sd(rating)/sqrt(n())) %>%
  filter(n >= 1000)  #use only categories with more than 1,000 ratings. 
head(categories) 

#Plot with error bars:
genr <- categories %>% mutate(genres = reorder(genres, avg))
genr %>%
  ggplot(aes(x = genres, y = avg, ymin = avg - 2*se, ymax = avg + 2*se)) + 
  geom_point() +
  geom_errorbar() 
  
# It appears that categories have a strong influence on ratings, 
# since the average per category ranges from 2 to above 4.5. 
# Furthermore, the small error bars show that the differences are
# highly significant. 

# Determine highest and lowest-rated categories:
 
#lowest 10  
genr %>% group_by(genres) %>% summarize(mu=mean(avg)) %>% top_n(-10,mu)

#highest10  
genr %>% group_by(genres) %>% summarize(mu=mean(avg)) %>% top_n(10,mu)

# The lowest rated is "Action|Children" at 2.04 and the highest is "Drama|Film-Noir|Romance" at 4.30.

# See if any single genre label, such as "Action," has an effect.
# To obtain the genre labels, we split the category strings on the
# "|" character.

 
# Extract all the different label possibilities
vector_of_labels <- unique(flatten(str_split(genr$genres,"\\Q|\\E")))

# Go through categories and see which label has the most ratings
head(vector_of_labels)
length(vector_of_labels)

# Determine which ones to pick by computing how many ratings appear for each individual genre.
sum_ratings_in_genre<- rep(0,length(vector_of_labels))
ratings_per_genre<- rep(0,length(vector_of_labels))
j=1
for (genre_name in vector_of_labels){ #loop over every label
  i=1 
  while(i<=nrow(genr)){ #loop over every category
    # add the ratings for this category to the running total in ratings_per_genre
    n_ratings<- ifelse(str_detect(genr$genres[i],genre_name),genr$n[i],0)
    ratings_per_genre[j] <- ratings_per_genre[j]+n_ratings
    
    i = i+1
  }
  #track the sum of the ratings so we can get an average later
  sum_ratings_in_genre[j] <- sum_ratings_in_genre[j]+ratings_per_genre[j]*genr$avg[j]
  j=j+1
}
#put results into a data frame
genres_df<- as.data.frame(cbind(vector_of_labels,ratings_per_genre))
genres_df
#which ones have over 2 million ratings?
genres_df%>%filter(ratings_per_genre>2000000) 
 
# The most frequently-rated genres are Action, Comedy, Drama, and Thriller, with over 2 million ratings each. For each of these 4, a binary variable, such as *action,* will be set to 1 if the genre contains the label "Action" and 0 otherwise. 
 
# Create four binary variables and assign 0 or 1 to each based on
# whether the category contains that genre (1) or not (0):
genr <- categories %>% 
  mutate(action = ifelse(str_detect(genres,"Action")==1,1,0))%>%
  mutate(comedy = ifelse(str_detect(genres,"Comedy")==1,1,0))%>%
  mutate(drama = ifelse(str_detect(genres,"Drama")==1,1,0))%>%
  mutate(thriller = ifelse(str_detect(genres,"Thriller")==1,1,0))
tail(genr)
# compute the mean rating and standard error for each individual genre:
genr%>%filter(action==1) %>%summarize(mean(avg),sd(avg)/sqrt(n()))
genr%>%filter(comedy==1) %>%summarize(mean(avg),sd(avg)/sqrt(n()))
genr%>%filter(drama==1) %>%summarize(mean(avg),sd(avg)/sqrt(n()))
genr%>%filter(thriller==1) %>%summarize(mean(avg),sd(avg)/sqrt(n()))
 
# Might the genres with the fewest ratings also be the lowest-rated? Let's find out. 

# Determine which genres have the least frequent ratings 
genres_df%>%filter(ratings_per_genre<200000) 
# The least-frequently rated genres are Western, Film Noir, IMAX and Documentary.
genr_least_common <- genr %>% 
  mutate(western= ifelse(str_detect(genres,"Western")==1,1,0))%>%
  mutate(noir = ifelse(str_detect(genres,"Film-Noir")==1,1,0))%>%
  mutate(imax = ifelse(str_detect(genres,"IMAX")==1,1,0))%>%
  mutate(doc = ifelse(str_detect(genres,"Documentary")==1,1,0))
#Do they also have the lowest ratings?
genr_least_common%>%filter(western==1) %>%summarize(n(), mean(avg),sd(avg)/sqrt(n()))
genr_least_common%>%filter(noir==1) %>%summarize(n(), mean(avg),sd(avg)/sqrt(n()))
genr_least_common%>%filter(imax==1) %>%summarize(n(), mean(avg),sd(avg)/sqrt(n()))
genr_least_common%>%filter(doc==1) %>%summarize(n(), mean(avg),sd(avg)/sqrt(n()))

# plot the average rating per genre versus the number of ratings for movies in the genre to see if there's a relationship overall:
 
genres_df<- genres_df %>% mutate(av_rating=as.numeric(sum_ratings_in_genre)/as.numeric(ratings_per_genre))
head(genres_df)
genres_df %>% 
  ggplot(aes(as.numeric(ratings_per_genre),av_rating)) + geom_point()+geom_smooth(method = "lm", se = FALSE)
 

# From the plot, there is no evidence that the number of ratings 
# in the genre predicts the rating. 
# So we won't use number of ratings in the genre as a predictor. 

# Mutate our dataset to include binary variables for each
# of the 4 highest-frequency labels. This expanded dataframe will
# be called *edx_plus*, and it will be used for training and 
# optimization from this point forward. 

edx_plus <-  edx %>%   mutate(action = ifelse(str_detect(genres,"Action")==1,1,0))%>%
  mutate(comedy = ifelse(str_detect(genres,"Comedy")==1,1,0))%>%
  mutate(drama = ifelse(str_detect(genres,"Drama")==1,1,0))%>%
  mutate(thriller = ifelse(str_detect(genres,"Thriller")==1,1,0))
names(edx_plus)

#######################################################
# Step 3: Modeling with the Training Set (edx_plus)
########################################################
y <- edx_plus$rating #desired outcome to predict

# Start by splitting edx_plus into train/test sets
# The training set is larger by a factor of 3
set.seed(2)
test_index <- createDataPartition(y, times = 1, p = 0.25, list = FALSE)
train_set <- edx_plus %>% slice(-test_index)
test_set <- edx_plus %>% slice(test_index)
dim(train_set) 
dim(test_set)

# Estimating the overall average rating: test vs. train
train_average <- train_set %>% summarize(pi=mean(rating)) %>% pull(pi)
train_average
test_set %>% summarize(pi=mean(rating)) %>% pull(pi)
# Consistency between test and train
# looks reasonable, as it should for such a large data set.

RMSE = function(predicted, actual){    #define RMSE function
  sqrt(mean((predicted - actual)^2))
}
# Let's create a "default" model and obtain a default value for the RMSE
# for comparison purposes. The default model will "predict"
# that all ratings are just equal to the average of the training set.

# use training set average to produce predictions
avg_vector<- rep(train_average,nrow(test_set)) 
default_model<- RMSE(avg_vector,test_set$rating) #test against the test set
default_model
# As expected, the RMSE using the average is terrible (>1), so 
# we will now test a model with predictors to see if we can do better. 

# Bias by movie ID
movie_avgs <- train_set %>% group_by(movieId) %>% 
  summarize(b_mov = mean(rating - train_average))

# Calculate RMSE on the test set
# You can only make predictions for movies you have in the 
# training set, so we get some NA in the predicted ratings
pred<- test_set %>% left_join(movie_avgs, by='movieId') %>%
  select(movieId,rating,b_mov)

pred_list_mov<- pred[!is.na(b_mov)] %>% #remove the NA's
  mutate(pred_rating=train_average+b_mov) #add movie bias

RMSE(pred_list_mov$pred_rating,pred_list_mov$rating) #test against the test set

# This is better (0.9440389)
# To correct for movies with low numbers of ratings, we introduce
# regularization with a tuning parameter, lambda.
# Use 5-fold cross validation to select the best lambda value
 
set.seed(1, sample.kind = "Rounding")  
indexes <- createResample(train_set$rating, 5) #create the 5 folds
ind1<- indexes$Resample1
ind2<- indexes$Resample2
ind3<- indexes$Resample3
ind4<- indexes$Resample4
ind5<- indexes$Resample5
# Create 5 test and training sets
train5<-train_set[c(ind1,ind2,ind3,ind4)]
test5<- train_set[ind5]
train1<-train_set[c(ind5,ind2,ind3,ind4)]
test1<- train_set[ind1]
train2<-train_set[c(ind1,ind5,ind3,ind4)]
test2<- train_set[ind2]
train3<-train_set[c(ind5,ind2,ind1,ind4)]
test3<- train_set[ind3]
train4<-train_set[c(ind5,ind2,ind1,ind3)]
test4<- train_set[ind4]

#free up some memory
remove(edx)
gc() 

# For each training set, we compute the error for each value of lambda.
# Using all 5 training sets, we average the error to find the lambda
# that minimizes the average error across all 5
lambdas <- seq(0.2,1,0.1) # tuning parameter

# Define a function that takes a training and test set and
# outputs a set of RSMEs for several choices of lambda
run_rmse<- function(training,testing){
sapply(lambdas, function(l){
  just_the_sum <- training %>% 
    group_by(movieId) %>% 
    summarize(s = sum(rating - train_average), n_i = n())
  predicted_rating <- testing %>% select(rating,movieId)%>%
    inner_join(just_the_sum, by='movieId') %>%
    mutate(b_i = s/(n_i+l))%>%  #compute the movie bias
    mutate(pred = train_average + b_i) #apply the bias
  rmse<- RMSE(predicted_rating$pred,predicted_rating$rating)
  return(rmse)
})}

#Apply the function to each test/training set to get 5 sets of output
rmse1<- run_rmse(train1,test1)
qplot(lambdas,rmse1) #check that range for lambda is reasonable
rmse2<- run_rmse(train2,test2)
rmse3<- run_rmse(train3,test3)
rmse4<- run_rmse(train4,test4)
rmse5<- run_rmse(train5,test5)

#calculate the average RMSE for each value of lambdas
rmse_list<- as.matrix(rmse1,rmse2,rmse3,rmse4,rmse5)
#  determine which one is the minimal one and select that lambda
l_best <- lambdas[which.min(rowMeans(rmse_list))]
l_best #<- 0.8
 
############ User Effects ###############################################
#Repeat the same cross-validation process to get a lambda value for user effects.
lambdas <- seq(2,11,1)
run_rmse<- function(training,testing){
  sapply(lambdas, function(l){
    sum_u <- training %>% 
      group_by(userId) %>% 
      summarize(s_u = sum(rating - train_average), n_u = n())
    predicted_rating <- testing %>% select(rating,userId)%>%
      inner_join(sum_u, by='userId') %>%
      mutate(b_u = s_u/(n_u+l))%>%
      mutate(pred = train_average + b_u) 
    rmse<- RMSE(predicted_rating$pred,predicted_rating$rating)
    return(rmse)
  })}
rmse1<- run_rmse(train1,test1)
qplot(lambdas,rmse1) #check that range for lambda is reasonable
rmse2<- run_rmse(train2,test2)
qplot(lambdas,rmse2) #check that range for lambda is reasonable
rmse3<- run_rmse(train3,test3)
rmse4<- run_rmse(train4,test4)
rmse5<- run_rmse(train5,test5)
 
#calculate the average RMSE for each value of lambdas
#  determine which one is the minimal one and select that lambda
rmse_list<- as.matrix(rmse1,rmse2,rmse3,rmse4,rmse5)
rmse_list
l_user<- lambdas[which.min(rowMeans(rmse_list))]
l_user # l_user = 4 minimizes the RMSE

# We obtain a lambda of 4.
# Now run these regularizations on the test set that wasn't used in 
# the cross-validation; use entire training set to compute the biases

sum_mov <- train_set %>%  #movie effects
  group_by(movieId) %>% 
  summarize(s_mov = sum(rating - train_average), n_i = n())

sum_u <- train_set %>%   #user effects
  left_join(sum_mov, by='movieId') %>%
  mutate(b_mov = s_mov/(n_i+l_best))%>%
  group_by(userId) %>% 
  summarize(s_u = sum(rating - train_average - b_mov), n_u = n())

# Start by modeling with movie effect only on the test set
predicted_rating <- test_set %>% select(rating,movieId)%>%
  left_join(sum_mov, by='movieId') %>%
  mutate(b_mov = s_mov/(n_i+l_best))%>%
  mutate(pred = train_average + b_mov)
predicted_rating<-predicted_rating[!is.na(pred)] 

#Compare predicted to actual ratings for the regularized effect of movieId
movie_reg<- RMSE(predicted_rating$pred,predicted_rating$rating)

movie_reg #RMSE = 0.9439934  

#########################################################################
# To reduce it further, we now implement user effects with regularization. 

# Model with movie effect and user effects
predicted_rating <- test_set %>% select(rating,movieId,userId)%>%
  left_join(sum_mov, by='movieId') %>%
  left_join(sum_u, by='userId') %>%
  mutate(b_mov = s_mov/(n_i+l_best))%>%
  mutate(b_u = s_u/(n_u+l_user))%>%
  mutate(pred = train_average + b_mov+b_u) 
predicted_rating<-predicted_rating[!is.na(pred)] #remove NA values

#Compare predicted ratings to actual ratings for the regularized effects
user_movie_reg<- RMSE(predicted_rating$pred,predicted_rating$rating)
user_movie_reg
## The RMSE is now 0.8662249. 
 
############ Genre effects ############################################### 

# Obtain individual user bias for drama genre using training set. 
# Start with the user and movie effects (already optimized)
predicted_rating <- train_set %>%
  left_join(sum_mov, by='movieId') %>%
  left_join(sum_u, by='userId') %>%
  mutate(b_mov = s_mov/(n_i+l_best))%>%
  mutate(b_u = s_u/(n_u+l_user))%>%
  select(userId,rating,b_u,b_mov,drama,action,thriller,comedy)

# every user has their own value of bias for each genre

#add in bd variable for drama bias 
drama_avgs <- predicted_rating %>% group_by(userId)   %>% 
     summarize(bd=mean(drama*(rating - train_average-b_mov-b_u)))

#add in ba variable for action bias
action_avgs <- predicted_rating %>% group_by(userId)   %>% 
  summarize(ba=mean(action*(rating - train_average-b_mov-b_u))) 

#add in bt variable for thriller bias
thriller_avgs <- predicted_rating %>% group_by(userId)   %>% 
  summarize(bt=mean(thriller*(rating - train_average-b_mov-b_u))) 

#add in bc variable for comedy bias
comedy_avgs <- predicted_rating %>% group_by(userId)   %>% 
  summarize(bc=mean(comedy*(rating - train_average-b_mov-b_u))) 

# Now test on the test set (still part of edx_plus, not the validation set)

# Start with user and movieId effects
predicted_rating_test <- test_set %>% 
  select(rating,movieId,userId,drama,action,thriller,comedy)%>%
  left_join(sum_mov, by='movieId') %>%
  left_join(sum_u, by='userId') %>%
  mutate(b_mov = s_mov/(n_i+l_best))%>%
  mutate(b_u = s_u/(n_u+l_user))%>%
  mutate(pred = train_average + b_mov+b_u)%>%
  select(rating,movieId,userId,drama,action,thriller,comedy,pred)
 #this prediction (pred) does not yet have genre effects
 
#Adjust the predictions for genre biases
#To save memory, I do this sequentially, mutating the prediction each time

adj_rating1 <- predicted_rating_test[!is.na(pred)]%>%  
  left_join(drama_avgs, by='userId')%>% 
  mutate(pred = pred+bd*drama)  #adds drama bias
head(adj_rating1) #check that bd is specific to the user
 
adj_rating2 <- adj_rating1  %>%
  select(rating,userId,movieId,comedy,action, pred,thriller)  %>%
  left_join(action_avgs, by='userId') %>% 
  mutate(pred = pred+ba*action)  #adds action bias
remove(adj_rating1) #free up memory
gc()

adj_rating3 <- adj_rating2 %>% 
  select(rating,userId,movieId,comedy,pred,thriller) %>%
  left_join(thriller_avgs, by='userId') %>% 
  mutate(pred = pred+bt*thriller)%>%  #adds thriller bias 
  select(rating,userId,movieId,comedy,pred)

head(adj_rating3)
adjusted_pred<- adj_rating3 %>% 
  left_join(comedy_avgs, by='userId') %>% 
  mutate(pred = pred+bc*comedy)%>%  #adds comedy bias 
  select(rating,userId,movieId,pred)

check_RMSE<- RMSE(adjusted_pred$pred,adjusted_pred$rating)
check_RMSE # We achieve an RMSE of 0.8603188 for the test set

# Let's do one more quick adjustment
# Round down predictions above 5
adjusted_pred<- adjusted_pred %>% mutate(pred=ifelse(pred>5,5,pred))
check_RMSE<- RMSE(adjusted_pred$pred,adjusted_pred$rating)
check_RMSE

# New RMSE is 0.8601547 
# This looks promising; it appears that selecting the top 4 genres
# to use for binary variables is enough. Having made this decision, 
# we are ready to validate our model.

#######################################################
# Step 4: Validation
########################################################

# Mutate validate data frame to pull out individual genres
# "validate_plus" includes binary variables: action, comedy, drama and thriller
validate_plus <- validation %>% 
  mutate(action = ifelse(str_detect(genres,"Action")==1,1,0)) %>%
  mutate(comedy = ifelse(str_detect(genres,"Comedy")==1,1,0)) %>%
  mutate(drama = ifelse(str_detect(genres,"Drama")==1,1,0)) %>%
  mutate(thriller = ifelse(str_detect(genres,"Thriller")==1,1,0))

gc() #clean up memory; this dataframe is big

# add in user and movie bias
predicted_rating_v <- validate_plus  %>% 
  select(rating,movieId,userId,drama,action,comedy,thriller) %>%
  left_join(sum_mov, by='movieId') %>%
  left_join(sum_u, by='userId') %>%
  mutate(b_mov = s_mov/(n_i+l_best)) %>%
  mutate(b_u = s_u/(n_u+l_user)) %>%
  mutate(pred = train_average + b_mov+b_u) %>%
  select(movieId,userId,pred,rating,drama,action,comedy,thriller) %>% na.omit()

# Let's see where we are at with just user and movie effects
user_movie_reg_v<- RMSE(predicted_rating_v$pred,predicted_rating_v$rating)

# We are at 0.8661695 without any genre effects.  
# Now adjust for genre bias by user.  This algorithm uses individual genre
# preferences of the user (action, thriller, comedy, drama), not the 
# aggregated categories (such has 'Action|Thriller')

 
adj_rating1 <- predicted_rating_v[!is.na(pred)]%>%  
  left_join(drama_avgs, by='userId')%>% 
  mutate(pred = pred+bd*drama)  #adds drama bias
head(adj_rating1) #check for plausibility

adj_rating2 <- adj_rating1  %>%
  select(rating,userId,movieId,comedy,action, pred,thriller)  %>%
  left_join(action_avgs, by='userId') %>% 
  mutate(pred = pred+ba*action)  #adds action bias
remove(adj_rating1) #free up memory
gc()

adj_rating3 <- adj_rating2 %>% 
  select(rating,userId,movieId,comedy,pred,thriller) %>%
  left_join(thriller_avgs, by='userId') %>% 
  mutate(pred = pred+bt*thriller)%>%  #adds thriller bias 
  select(rating,userId,movieId,comedy,pred)

head(adj_rating3)
adjusted_pred<- adj_rating3 %>% 
  left_join(comedy_avgs, by='userId') %>% 
  mutate(pred = pred+bc*comedy)%>%  #adds comedy bias 
  select(rating,userId,movieId,pred)

adjusted_pred<- adjusted_pred %>% mutate(pred=ifelse(pred>5,5,pred))
final_RMSE<- RMSE(adjusted_pred$pred,adjusted_pred$rating)
final_RMSE

## For comparison, we will also re-run the default model, 
## movie-only model, and movie/user effects model on the validation set

#re-run the default model on the validation set 
avg_vector_v<- rep(train_average,nrow(validation)) 
default_model_v<- RMSE(avg_vector_v,validation$rating) #test against the test set
default_model_v

# Re-run movie effect on the validation set
predicted_rating <- validation %>% select(rating,movieId)%>%
  left_join(sum_mov, by='movieId') %>%
  mutate(b_mov = s_mov/(n_i+l_best))%>%
  mutate(pred = train_average + b_mov)
predicted_rating<-predicted_rating[!is.na(pred)] 

#Compare predicted to actual ratings for the regularized effect of movieId
movie_reg_v<- RMSE(predicted_rating$pred,predicted_rating$rating)
movie_reg_v    


# Compare all models in a table:
# 1: default model (no bias)
# 2: movie bias only (with regularization)
# 3: movie bias and user bias (with regularization)
# 4: final model: movie & user bias (with regularization) plus genre  
results <-  as.table(cbind(default_model_v, movie_reg_v, user_movie_reg_v, final_RMSE))
row.names(results)<-  'RMSE' 
colnames(results)<- c('Default ',' Movie ', ' User+Movie ', ' Final Model ')
results 
