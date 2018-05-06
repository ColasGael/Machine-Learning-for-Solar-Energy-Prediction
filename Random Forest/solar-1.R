library(randomForest)
library(visreg)
library(gbm)
all <- read.csv("hourly-weather-dataset_random-order.csv") 
all <- subset(all, select = -X)
all.train <- filter(all, Random < 0.8)
all.dev <- subset(all , Random > .8 & Random < 0.9)
trial <- gbm.step(data=all.train, gbm.x = 4:11, gbm.y = 12, family = "laplace", tree.complexity = 15, 
                            learning.rate = 0.2, bag.fraction = 0.75)

# We have to choose a family, which represents a loss function. Options: "gaussian" (for minimizing squared error),
# "laplace" (for minimizing absolute loss), "bernoulli" (logistic regression for 0-1 outcomes),
# "poisson" (count outcomes; requires the response to be a positive integer)


names(trial)
summary(trial)


trial.pred <- predict.gbm(trial, all.dev, n.trees = trial$gbm.call$best.trees, type="response")
calc.deviance(obs=all.dev$Solar.energy, pred=trial.pred, calc.mean=TRUE)
df <- as.data.frame(cbind(all.dev$Solar.energy, trial.pred))
df.non <- filter( df, df$V1 > 0)

cor(df.non$V1, df.non$trial.pred)
mean(abs((df.non$V1 - df.non$trial.pred)/df.non$V1))

mean((df.non$V1 - df.non$trial.pred)^2)
median((df.non$V1 - df.non$trial.pred)^2)

mean((df$V1 - df$trial.pred)^2)
median((df$V1 - df$trial.pred)^2)

mean((df.non$V1 - df.non$trial.pred)/df.non$V1)
plot(seq(1:756), df$V1, col = "red", type = "l")
lines(seq(1:756), df$trial.pred, col = "black")




## training data:

train <- read.csv("weather_train.csv", sep = ";", header = F)
colnames(train) <- c("Hour", "Day", "Month", "Year", "Cloud.coverage", "Visibility", "Temperature", "Dew.point",
                     "Relative.humidity","Wind.speed", "Station.pressure", "Altimeter", "Solar.energy")

train.model <- gbm.step(data= train, gbm.x = 5:12, gbm.y = 13, family = "laplace", tree.complexity = 15, 
                                 learning.rate = 0.5, bag.fraction = 0.65)
names(train.model)
summary(train.model)

best.iter <- gbm.perf(train.model,method="test")
print(best.iter)


best.iter <- gbm.perf(train.model,method="cv")
print(best.iter)


summary(train.model,n.trees=1)         # based on the first tree
summary(train.model,n.trees=best.iter) # based on the estimated best number of trees


## create correlations plot:

library(plyr)
cor <- read.csv("pcaCorrelation.csv")

cor$name <- mapvalues(cor$name, 
                               from=c("Cloud.coverage" , "Relative.humidity" , "Wind.speed" ,  "Station.pressure"), 
                               to=c("Cloud coverage", 
                                    "Relative humidity", "Wind speed", "Station pressure"))

ggplot(data = cor, aes(x = name, y = cor, fill = type)) +
  geom_bar(stat="identity", width = 0.8) + facet_grid(type~.) + theme_light() +
  ylab("Relative influence (%)") +  xlab("Feature") +
  scale_fill_manual(values=c("firebrick2", "dodgerblue2")) + theme(axis.text.x=element_text(angle=90,hjust=1))

ggsave("corr-v2.pdf", width = 4.5, height = 5.5)



test <- read.csv("weather_test.csv", sep = ";", header = F)
colnames(test) <- c("Hour", "Day", "Month", "Year", "Cloud.coverage", "Visibility", "Temperature", "Dew.point",
                     "Relative.humidity","Wind.speed", "Station.pressure", "Altimeter", "Solar.energy")


trial.pred <- predict.gbm(train.model, test, n.trees = train.model$gbm.call$best.trees, type="response")
calc.deviance(obs=test$Solar.energy, pred=trial.pred, calc.mean=TRUE)
df <- as.data.frame(cbind(test$Solar.energy, trial.pred))

colnames(df) <- c("Solar.energy", "Predictions")
test.all <- test
test.all$Predictions <- df$Predictions

test.all <- test.all %>% mutate(abs.rel.error = abs((Solar.energy-Predictions)/Solar.energy))

test.all <- test.all %>% mutate(LSM = ((Solar.energy-Predictions)^2))

test.all.hour <- subset(test.all, Hour %in% c(8, 12, 16))
test.all.hour <- subset(test.all.hour, select = c(Hour, Day, Month, Year, abs.rel.error))
test.all.hour$type <- "GBM"
colnames(test.all.hour) <- c("Hour" , "Day"  , "Month", "Year" , "Error", "type")
## PCA error:

PCA.error <- read.csv("error_pca5_all.csv")

PCA.error.all <- rbind(PCA.error, test.all.hour)

PCA.error.all$date <- as.Date(with(PCA.error.all, paste(Month, Day, Year,sep="-")), "%m-%d-%Y")
PCA.error.all$Hour <- mapvalues(PCA.error.all$Hour , 
                      from=c("8", "12", "16"), to = c("8AM", "12PM", "4PM"))
PCA.error.all$Hour_f = factor(PCA.error.all$Hour, levels=c("8AM", "12PM", "4PM"))



ggplot(PCA.error.all, aes(x = date, y = Error, color = type)) + geom_line(size=.7) + theme_light() +
  labs(x = "Date", y = "Absolute relative error") + facet_grid(Hour_f~., scales = "free_y") + 
scale_x_date(date_labels = "%m-%d-%Y", date_breaks = "1 month") + 
  theme(axis.text.x=element_text(angle=90, hjust=1)) +
  scale_color_manual(values=c("dodgerblue2", "darkslategray2", "firebrick2"))

ggsave("time-series-error-v1.pdf", width = 6.5, height = 5.5)
  


