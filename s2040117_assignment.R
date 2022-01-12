library(readr)
library(dplyr)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(tidyr)

set.seed(100)

### Data summary & pre-processing

# loading dataset
dataset <- read_csv("covid-19 symptoms dataset.csv")

# structure, dimension of dataset
str(dataset)

# statistical summary of dataset
summary(dataset)       # found categorical variables set as numeric format

# no na/empty cells detected
colSums(is.na(dataset))

# change numeric format into factors format at categorical variables
dataset2 <- dataset %>%
  mutate(bodyPain = factor(bodyPain))%>%
  mutate(runnyNose = factor(runnyNose))%>%
  mutate(diffBreath = factor(diffBreath))%>%
  mutate(infectionProb = factor(infectionProb))

# double check the processed dataset
str(dataset2)
summary(dataset2)


### 70-30 train-test split
ncovid <- dataset2 %>% filter(infectionProb==0)
covid <- dataset2 %>% filter(infectionProb==1)

sel_covid <- sort(sample(nrow(covid),size=0.7*nrow(covid)))
sel_ncovid <- sort(sample(nrow(ncovid),size=0.7*nrow(ncovid)))

train <- rbind(covid[sel_covid,],ncovid[sel_ncovid,])
test <- rbind(covid[-sel_covid,],ncovid[-sel_ncovid,])

# check distribution of target variable in both dataset.
prop.table(table(train$infectionProb))
prop.table(table(test$infectionProb))


### Decision tree (rpart) modelling

##Tune complexity parameter of the rpart tree control

fit_df <- data.frame(accuracy=double())

for (i in 1:50){
  fit <- rpart(infectionProb~.,data = train, 
               method = 'class',
               control = rpart.control(cp=i*0.001))
  
  prediction <- predict(fit,test,type='class')

  cfmat <-table(prediction,actual=test$infectionProb)

  accuracy_test <- sum(diag(cfmat))/sum(cfmat)
  
  df<-data.frame(accuracy=round(accuracy_test*100,2))
  
  fit_df <- rbind(fit_df,df)
}

# plot against cp range
ggplot(fit_df,aes(x=1:nrow(fit_df),y=accuracy)) +
  geom_line() +
  labs(title = "Accuracy of rpart function against cp value",
       x="cp = 0.001*n")

# cp parameter for highest accuracy performance
j<-which.max(fit_df$accuracy)

## rpart modelling with best cp parameter, cross validation = 10

fit2 <- rpart(infectionProb~.,data=train,
             method='class',
             control=rpart.control(xval=10,cp=0.001*j))

# plot decision tree figure
rpart.plot(fit2)


## Modelling

# prediction using test data
prediction <- predict(fit2,test,type='class')

# confusion matrix
cfmat <-table(prediction,actual=test$infectionProb)

# accuracy/sensitivity/specificity/precision
accuracy_test <- sum(diag(cfmat))/sum(cfmat)
sensitivity_test <- cfmat[1,1]/sum(cfmat[,1])
specificity_test <- cfmat[2,2]/sum(cfmat[,2])
precision_test <- cfmat[1,1]/sum(cfmat[1,])

dt_perf<-data.frame(Model='DT',
                    accuracy=round(accuracy_test*100,2),
                    sensitivity=round(sensitivity_test*100,2),
                    specificity=round(specificity_test*100,2),
                    precision=round(precision_test*100,2))

# decision tree performance measure
dt_perf

# variable importance based on rpart function

fit2$variable.importance


### Random Forest modelling

## Tuning ntree
model <- randomForest(infectionProb~.,train,ntree=3000)
model

# extract error rates data
err <- data.frame(
  Trees = rep(1:nrow(model$err.rate),times=3),
  Type = rep(c("OOB","non-covid","covid"),each=nrow(model$err.rate)),
  Error = c(model$err.rate[,"OOB"],model$err.rate[,"0"],model$err.rate[,"1"]))

# plot error rates against ntree
ggplot(err,aes(Trees,Error))+
  geom_line(aes(color=Type),lwd=1) +
  ggtitle("Error rates respective to ntree parameter")


## Tuning mtry with ntree = 1500
bestmtry <- tuneRF(train[,-6],
                   as.factor(train$infectionProb),
                   mtryStart=2,
                   stepFactor=2,
                   improve=1e-5,
                   ntreeTry=1500)
bestmtry

## Modelling
model <- randomForest(infectionProb~.,train,importance=TRUE,mtry=2,ntree=1500)
prediction <- predict(model,test,method='class')

# performance measures
cfmat <- table(prediction,actual=test$infectionProb)
accuracy_test <- sum(diag(cfmat))/sum(cfmat)
sensitivity_test <- cfmat[1,1]/sum(cfmat[,1])
specificity_test <- cfmat[2,2]/sum(cfmat[,2])
precision_test <- cfmat[1,1]/sum(cfmat[1,])

rf_perf<-data.frame(Model='RF',
                    accuracy=round(accuracy_test*100,2),
                    sensitivity=round(sensitivity_test*100,2),
                    specificity=round(specificity_test*100,2),
                    precision=round(precision_test*100,2))

# randomforest model performance measures
rf_perf

# sorted variable importance based on mean decrease Gini value
ranked_importance <- data.frame(model$importance) %>%
  arrange(desc(MeanDecreaseGini))

ranked_importance



### Logistic regression

lrmodel <- glm(infectionProb~.,train,family = 'binomial')
summary(lrmodel)

predictlm <- predict(lrmodel,test,type='response')
prediction <- ifelse(predictlm>0.5,1,0)

cfmat <- table(prediction,actual=test$infectionProb)
accuracy_test <- sum(diag(cfmat))/sum(cfmat)
sensitivity_test <- cfmat[1,1]/sum(cfmat[,1])
specificity_test <- cfmat[2,2]/sum(cfmat[,2])
precision_test <- cfmat[1,1]/sum(cfmat[1,])

lr_perf<-data.frame(Model='LR',
                    accuracy=round(accuracy_test*100,2),
                    sensitivity=round(sensitivity_test*100,2),
                    specificity=round(specificity_test*100,2),
                    precision=round(precision_test*100,2))

# performance measures
lr_perf

# ranked odd ratio
odd_ratio <- exp(lrmodel$coefficients[-1])
rank<-data.frame(oddratio=sort(odd_ratio,decreasing = TRUE))
rank



### Overall result
overall <- rbind(dt_perf,rf_perf,lr_perf)
overall

overall <- overall %>%
  mutate(Model=factor(Model,levels=c('DT','LR','RF'))) %>%
  mutate('F1_score'=round(2*precision*sensitivity/(precision+sensitivity),2)) %>%
  gather(key='Metrics',value='Measure',accuracy,sensitivity,specificity,precision,F1_score) %>%
  mutate(Metrics=factor(Metrics,levels=c('accuracy','sensitivity','specificity','precision','F1_score'))) %>%
  arrange(Model)

ggplot(overall,aes(x=Metrics,y=Measure,fill=Model))+
  geom_bar(stat='identity',position='dodge')+
  scale_fill_manual(values=c('indianred1','seagreen2','cornflowerblue'))+
  labs(x="",
       y="Measures(%)",
       title="Evaluation metrics respective to Models")+
  geom_text(aes(label=Measure),
            size=3.5,
            vjust=-.5,
            position=position_dodge(.9))+
  theme(text = element_text(size=13),
        axis.text.y = element_text(color='black'),
        axis.text.x = element_text(color='black',face='bold'))
