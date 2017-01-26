#Titanic Competition

##Background
The Titanic competition is provided by Kaggle which missioned to predict the passengers' survival during the Titanic tragedy. The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

##Main Goals (for myself)
1. Conduct EDA and visualize the data
2. Build ML models and enhance predictivity

##Data Sets
The data stes are provided by Kaggle. The total amount of rows are 1309, and we separate 891 of them as training data and other 418 of them as testing data. The variable descripions are as below (provided by Kaggle):
```
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)

SPECIAL NOTES:
Pclass is a proxy for socio-economic status (SES)
 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

Age is in Years; Fractional if Age less than One (1)
 If the Age is Estimated, it is in the form xx.5

With respect to the family relation variables (i.e. sibsp and parch)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
Parent:   Mother or Father of Passenger Aboard Titanic
Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations.
```

##Settings
Load in new packages for future usage
```
install.packages("rpart") #Recursive Partitioning and Regression Trees
install.packages('rattle') #Plotting rpart
install.packages('rpart.plot')
install.packages('RColorBrewer')
install.packages('randomForest')
install.packages('party')
```

##Getting the Data Into R
```
test_data <- read.csv("test.csv")
train_data <- read.csv("train.csv")
```

##EDA
###Data structure
```
> str(train_data)
'data.frame':	891 obs. of  16 variables:
 $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
 $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
 $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
 $ Name       : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
 $ Sex        : Factor w/ 2 levels "female","male": 2 1 1 1 2 2 2 2 1 1 ...
 $ Age        : num  22 38 26 35 35 ...
 $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
 $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
 $ Ticket     : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
 $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
 $ Cabin      : Factor w/ 187 levels "","A10","A14",..: 1 83 1 57 1 1 131 1 1 1 ...
 $ Embarked   : Factor w/ 4 levels "","C","Q","S": 4 2 4 4 4 3 4 4 4 2 ...
 $ Has_Cabin  : int  0 1 0 1 0 0 1 0 0 0 ...
 $ Title      : Factor w/ 11 levels "Col","Dr","Lady",..: 7 8 6 8 7 7 7 5 8 8 ...
 $ Family_Size: num  2 2 1 2 1 1 1 5 3 2 ...
 $ Surname    : chr  "Braund" "Cumings" "Heikkinen" "Futrelle" ...
 ```
 ###Missing data
```
missmap(df.train, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)
```
![alt text](https://github.com/LeoWongTaiwan/Machine-Learning/blob/master/Titanic%20Competition/Figures/Missing%20Data.jpg)

###Survival Rate
About 60% of passengers survived
```
> round(prop.table(table(train_data$Survived)),4)
     0      1 
0.6162 0.3838 
```
###Sex vs Survival
```
GG <- ggplot(data=train_data)
GG_Avg_Line <- function(aes){GG+geom_bar(aes,position = "fill")+
    geom_hline(yintercept=0.3838, linetype="dashed")+
    geom_text(aes(0,0.3838,label = "Average Survival Rate = 0.3838", vjust = 2,hjust=0),
              family="BL",color="#333333")}
```
####Number of Passengers 
```
GG+geom_bar(aes(x=Sex,fill=Sex))
```
![alt text](https://github.com/LeoWongTaiwan/Machine-Learning/blob/master/Titanic%20Competition/Figures/Number%20of%20Passengers.png)

####Survival Rate of Different Sex
```
> round(prop.table(table(train_data$Sex,train_data$Survived),1),4) #sex row %   
              0      1
  female 0.2580 0.7420
  male   0.8111 0.1889
```
```
mosaicplot(train_data$Sex ~ train_data$Survived, 
           main="Passenger Fate by Sex", shade=FALSE, 
           color=TRUE, xlab="Sex", ylab="Survived")
```
![alt text](https://github.com/LeoWongTaiwan/Machine-Learning/blob/master/Titanic%20Competition/Figures/Passenger%20Fate%20by%20Sex.png)

###Pclass vs Survival
####Number of Passengers 
![alt text](https://github.com/LeoWongTaiwan/Machine-Learning/blob/master/Titanic%20Competition/Figures/Number%20of%20Passengers%20Pclass.png)

####Survival Rate of Different Pclass
```
> round(prop.table(table(train_data$Pclass,train_data$Survived),1),4) #Pclass row %
         0      1
  1 0.3704 0.6296
  2 0.5272 0.4728
  3 0.7576 0.2424
```
![alt text](https://github.com/LeoWongTaiwan/Machine-Learning/blob/master/Titanic%20Competition/Figures/Passenger%20Fate%20by%20Traveling%20Class.png)

###SibSp vs Survival
####Number of Passengers 
![alt text](https://github.com/LeoWongTaiwan/Machine-Learning/blob/master/Titanic%20Competition/Figures/Number%20of%20Passengers%20SibSp.png)

####Survival Rate of Different SibSp
![alt text](https://github.com/LeoWongTaiwan/Machine-Learning/blob/master/Titanic%20Competition/Figures/Number%20of%20Passengers%20SibSp.png)

###Parch vs Survival
```
> round(prop.table(table(train_data$Parch,train_data$Survived),1),4) #Parch row %
         0      1
  0 0.6563 0.3437
  1 0.4492 0.5508
  2 0.5000 0.5000
  3 0.4000 0.6000
  4 1.0000 0.0000
  5 0.8000 0.2000
  6 1.0000 0.0000
```
####Number of Passengers 
![alt text](https://github.com/LeoWongTaiwan/Machine-Learning/blob/master/Titanic%20Competition/Figures/Number%20of%20Passengers%20Parch.png)

####Survival Rate of Different Parch
![alt text](https://github.com/LeoWongTaiwan/Machine-Learning/blob/master/Titanic%20Competition/Figures/Survival%20Rate%20of%20Different%20SibSp.png)

###The following EDA will be shown as codes
```
##Embarked vs Survival
mosaicplot(train_data$Embarked ~ train_data$Survived, 
           main="Passenger Fate by Port of Embarkation", shade=FALSE, 
           color=TRUE, xlab="Embarked", ylab="Survived")

GG+geom_bar(aes(x=Embarked,fill=Embarked)) #Plot

GG_Avg_Line(aes(x=factor(Embarked),fill=factor(Survived)))

table(train_data$Embarked,train_data$Survived)
round(prop.table(table(train_data$Embarked,train_data$Survived)),4) #all %
round(prop.table(table(train_data$Embarked,train_data$Survived),1),4) #Parch row %
##Check Fare
ggplot(data=train_data,aes(x=factor(0),y=Fare))+  #Fare Distribution
  geom_boxplot()+ggtitle("Fare Distribution")+
  stat_summary(fun.y = mean, colour="#F8766D", geom="point", size=3,show.legend = FALSE)

GG+geom_histogram(aes(x=Fare,fill=factor(Survived)),binwidth=10,position = "fill",stats="count")+
  geom_hline(yintercept=0.3838, linetype="dashed")+ 
  scale_y_continuous(breaks = sort(c(seq(min(train_data$Survived), max(train_data$Survived), length.out=5), 0.3838)))+
  geom_text(aes(0,0.3838,label = "Average Survival Rate = 0.3838", vjust = 2,hjust=0),family="BL",color="#333333")

summary(train_data$Fare)

GG+geom_histogram(aes(x=Fare,y=..count..),binwidth=5)
train_data[c(which(train_data$Fare<=10)),]$Survived #Random trying
##Check Age
summary(train_data$Age)

ggplot(data=train_data, aes(x=factor(0),y=Age))+ 
  geom_boxplot(size=0.5)

GG+geom_histogram(aes(x=Age,fill=factor(Survived)),binwidth=5,position = "fill",stats="count")+
  geom_hline(yintercept=0.3838, linetype="dashed")+ 
  scale_y_continuous(breaks = sort(c(seq(min(train_data$Survived), max(train_data$Survived), length.out=5), 0.3838)))+
  geom_text(aes(0,0.3838,label = "Average Survival Rate = 0.3838", vjust = 2,hjust=0),family="BL",color="#333333")

boxplot(train_data$Age ~ train_data$Survived, 
        main="Passenger Fate by Age",
        xlab="Survived", ylab="Age")

##Check Cabin
mosaicplot(train_data$Has_Cabin ~ train_data$Survived, 
           main="Passenger Fate by Has Cabin or Not", shade=FALSE, 
           color=TRUE, xlab="Has Cabin", ylab="Survived")

summary(train_data$Has_Cabin)

GG+geom_bar(aes(x=Has_Cabin,fill=Has_Cabin)) #Plot
GG_Avg_Line(aes(x=Has_Cabin,fill=Survived)) #Plot Survival Rate
table(train_data$Parch,train_data$Survived)
round(prop.table(table(train_data$Parch,train_data$Survived)),4) #all %
round(prop.table(table(train_data$Parch,train_data$Survived),1),4) #Parch row %

##Check Age vs Sex
boxplot(train_data$Age ~ train_data$Sex, 
        main="Passenger Sex by Age",
        xlab="Sex", ylab="Age")

##Check Age vs Pclass
boxplot(train_data$Age ~ train_data$Pclass, 
        main="Passenger Pclass by Age",
        xlab="Pclass", ylab="Age")

##Check Age vs SibSp
boxplot(train_data$Age ~ train_data$SibSp, 
        main="Passenger SibSp by Age",
        xlab="SibSp", ylab="Age")

##Check Age vs Parch
boxplot(train_data$Age ~ train_data$Parch, 
        main="Passenger Parch by Age",
        xlab="Parch", ylab="Age")

##Check Age vs Parch
boxplot(train_data$Age ~ train_data$Parch, 
        main="Passenger Parch by Age",
        xlab="Parch", ylab="Age")


##Check Age vs Fare
plot(train_data$Age ~ train_data$Fare, 
        main="Passenger Fare by Age",
        xlab="Fare", ylab="Age")

##Check Age vs Embarked
boxplot(train_data$Age ~ train_data$Embarked, 
     main="Passenger Embarked by Age",
     xlab="Embarked", ylab="Age")

##Check Age vs Has_Cabin
boxplot(train_data$Age ~ train_data$Has_Cabin, 
     main="Passenger Has_Cabin by Age",
     xlab="Has_Cabin", ylab="Age")
```


##Data Transforming




![alt text]()
![alt text]()
![alt text]()
![alt text]()
