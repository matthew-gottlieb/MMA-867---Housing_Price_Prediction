library(tidyverse)
library(glmnet)
library(caret)
library(car)
library(readxl)
library(scales)
library(correlationfunnel)
library(tidyr)
library(recipes)



train <- read.csv('C:/Users/Matth/Desktop/Matthew/Smith MMA/MMA 867 - Predictive Modelling/Assignments/Individual Assignments/train.csv',
                  stringsAsFactors = F)

sale_price <- as.data.frame(train$SalePrice)

test <- read.csv('C:/Users/Matth/Desktop/Matthew/Smith MMA/MMA 867 - Predictive Modelling/Assignments/Individual Assignments/test.csv',
                 stringsAsFactors = F)

test_sp <- data.frame (SalePrice  = as.numeric(NA)) #Creating an empty column for Sales in test; this way our columns are the same length

test <- cbind(test,test_sp)
sale_price_test <- as.data.frame(test$SalePrice)


#Combining for data cleaning purposes
combined_df <- rbind(select(train, MSSubClass:SaleCondition),
                     select(test,MSSubClass:SaleCondition))


#Checking Distribtuion of Sales Price
qqnorm(train$SalePrice) #Looking skewed; let's also look at the distribution below

ggplot(train, aes(x = SalePrice))+
  geom_histogram(bins = 45, color="darkblue", fill="lightblue")+
  scale_x_continuous(breaks= seq(0, 800000, by=100000), labels = comma)+
  geom_vline(aes(xintercept=mean(SalePrice)),color="blue", linetype="dashed", size=1)+
  labs(title = 'Sales Price Distribution, Ames Iowa')+
  theme(plot.title = element_text(hjust = 0.5))


#Now lets look for correlations in the data
correlation <- select_if(train, is.numeric) %>%
  correlate(SalePrice)%>%
  arrange(desc(correlation))
#Will fix in Recipe data pre processing


#Data Prep

#Lets now check for NA values by column
missing_vals <- map_df(combined_df, function(x) sum(is.na(x))) %>%
  gather('Variable', 'Quantity') %>%
  arrange(desc(Quantity)) %>%
  mutate(`%Missing` = Quantity / 2919)

#Feature Engineering
#Adding in age of house; If remodelled can be considered like a reset for the house if terms of age
#If not remodelled, YearRemodAdd defaults to same value as YearBuilt
combined_df$age <- as.numeric(combined_df$YrSold - combined_df$YearRemodAdd)

#Also adding in if it is a new house
combined_df$new <- ifelse(combined_df$YrSold == combined_df$YearBuilt,1,0)

#Adding in dummy variables relating to quality of the house
combined_df$good_qual <- ifelse(combined_df$OverallQual >= 7,1,0)
combined_df$good_cond <- ifelse(combined_df$OverallCond >= 7,1,0)

combined_df$total_sqft <- combined_df$GrLivArea + combined_df$TotalBsmtSF

combined_df$total_bathrooms <- combined_df$FullBath + combined_df$HalfBath

combined_df$porch_area <- combined_df$WoodDeckSF + combined_df$OpenPorchSF + combined_df$EnclosedPorch + combined_df$X3SsnPorch + combined_df$ScreenPorch

#Changing Subclass & Dates to Factors so they are not included as numeric
summary(combined_df$MSSubClass) #Also needs to be changed to factor
combined_df$MSSubClass <- as.character(combined_df$MSSubClass)

summary(combined_df$YrSold) #change to a factor
combined_df$YrSold <- as.character(combined_df$YrSold)

summary(combined_df$MoSold) #change to a factor
combined_df$MoSold <- as.character(combined_df$MoSold)

summary(combined_df$YearBuilt) #change to a factor
combined_df$YearBuilt <- as.character(combined_df$YearBuilt)

summary(combined_df$GarageYrBlt) #change to a factor ***HAS NA VALUES***
combined_df$GarageYrBlt <- as.character(combined_df$GarageYrBlt) 

summary(combined_df$YearRemodAdd) 
combined_df$YearRemodAdd <- as.character(combined_df$YearRemodAdd)

str(combined_df)

#Splitting Combined DF into numeric dataset; will combine back later
numeric_df <- select_if(combined_df, is.numeric)
num_missing <- as.data.frame(colSums(is.na(numeric_df)))

#Splitting Combined DF into character/factor dataset; will combine back later
factor_df <- select_if(combined_df, is.character)%>%
  mutate_if(is.character, factor)

str(factor_df)

new_df <- cbind(numeric_df, factor_df)
str(new_df)

train <- new_df[1:1460,]
train <- cbind(sale_price, train)
colnames(train)[colnames(train) == 'train$SalePrice'] <- 'SalePrice'


test <- new_df[1461:2919,]
test <- cbind(sale_price_test, test)
colnames(test)[colnames(test) == 'test$SalePrice'] <- 'SalePrice'

#1st step - defining what we want to predict
model_recipe <- recipe(SalePrice ~., data = train)
summary(model_recipe)

#2nd Step - Writing our preprocessing steps
model_recipe_steps <- recipe(SalePrice ~ ., data = train) %>% 
  step_knnimpute(all_predictors()) %>%
  step_dummy(all_predictors(), -all_numeric()) %>%
  step_interact(terms = ~ starts_with('Neighborhood'):starts_with('OverallQual'))%>%
  step_interact(terms = ~ OverallQual:starts_with('MSSub'):starts_with('MSZon'))%>%
  step_BoxCox(all_predictors()) %>%
  step_center(all_predictors())  %>%
  step_scale(all_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_log(all_outcomes()) %>%
  check_missing(all_predictors())

#3rd Step - Prepping Recipe
prepped_recipe <- prep(model_recipe_steps, training = train)

#4th & Final Step; baking the recipe for both train & test
df_train_preprocessed <- bake(prepped_recipe, train) 
df_test_preprocessed <- bake(prepped_recipe, new_data =test) 

#Checking Sales Price for Normality
plot(df_train_preprocessed$SalePrice)
qqnorm(df_train_preprocessed$SalePrice) #Looks much better

ggplot(df_train_preprocessed, aes(x = SalePrice))+
  geom_histogram(bins = 45, color="darkblue", fill="lightblue")+
  geom_vline(aes(xintercept=mean(SalePrice)),color="blue", linetype="dashed", size=1)+
  labs(title = 'Sales Price Distribution, Ames Iowa',
       x = 'log Sale Price')+
  theme(plot.title = element_text(hjust = 0.5))




#Predictive Modelling
which( colnames(df_train_preprocessed)=="SalePrice" )
which( colnames(df_test_preprocessed)=="SalePrice" )

X_train <- df_train_preprocessed[,-38]
X_test <- df_test_preprocessed[,-38]
y <- df_train_preprocessed$SalePrice

#Creating listo of lambda values
lambda <- seq(0.001,0.1,by = 0.0005)

tr <- trainControl(method = 'repeatedcv', number = 10, repeats = 5, savePredictions = "all")

#LASSO Regression
lasso_fit <- train(x = X_train, y = y, method = 'glmnet',
                   metric = 'RMSE',
                   trControl = tr,
                   tuneGrid = expand.grid(alpha = 1, lambda = lambda))



lasso_fit$results

mean(lasso_fit$resample$RMSE)
mean(lasso_fit$resample$Rsquared)
coef(lasso_fit$finalModel, lasso_fit$bestTune$lambda) #Showing coefficients that were picked
lasso_fit$bestTune #Displays optimal Lambda

lasso_prediction <- exp(predict(lasso_fit, newdata = X_test))






#Ridge Regression
ridge_fit <- train(x = X_train, y = y, method = 'glmnet',
                   metric = 'RMSE',
                   trControl = tr,
                   tuneGrid = expand.grid(alpha = 0, lambda = lambda))


ridge_fit$results

mean(ridge_fit$resample$RMSE)
mean(ridge_fit$resample$Rsquared)
coef(ridge_fit$finalModel, lasso_fit$bestTune$lambda) #Showing coefficients that were picked
ridge_fit$bestTune #Displays optimal Lambda



#------------------------------------------------------------------------------
#Writing final predictions
solution <- data.frame(Id=as.integer(rownames(test)),SalePrice=lasso_prediction)
write.csv(solution,"lasso_preds.csv",row.names=FALSE)


