#Titanic Survival Prediction

##Background 
The Titanic challenge is a competition from Kaggle missioned to build predicitive models for passengers' survival. The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

##Main Goals (for myself)
1. Conduct Exploratory Data Analysis on the data and visualize interesting relationships between different columns.
2. Learn and use different ML models to boost accuracy of the prediction.

##Data
There are totally 1309 passengers that are recorded in the data set. We separatde 891 of them to be the training data and the rest 418 of them to be the testing data. All the datas are provided by Kaggle.The description od columns are as below (from Kaggle):
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

## Basic Settings
Load additional packages.
```
install.packages("rpart") #Recursive Partitioning and Regression Trees
install.packages('rattle') #Plotting rpart
install.packages('rpart.plot')
install.packages('RColorBrewer')
install.packages('randomForest')
install.packages('party')
```

## Getting the Data Into R
Load in the data sets and rename them.
```
test_data <- read.csv("test.csv")
train_data <- read.csv("train.csv")
```
Lets see columns do we have
```
> colnames(train_data)
 [1] "PassengerId" "Survived"    "Pclass"      "Name"        "Sex"         "Age"         "SibSp"      
 [8] "Parch"       "Ticket"      "Fare"        "Cabin"       "Embarked"    
> colnames(test_data)
 [1] "PassengerId" "Survived"    "Pclass"      "Name"        "Sex"         "Age"         "SibSp"      
 [8] "Parch"       "Ticket"      "Fare"        "Cabin"       "Embarked"  
 ```

##Data Cleaing and Transforming
Data summary
Missing data in train and test data

1. Train <- Pclass:full, Sex:full, Age: 177 NAs, SibSp:full, Parch:full, Fare:1 NA, Embarked:full, Has_Cabin:full
2. Test <- Pclass:full, Sex:full, Age: 86 NAs, SibSp:full, Parch:full, Fare:1 NA, Embarked:full, Has_Cabin:full
```
> summary(train_data)
  PassengerId       Survived          Pclass          Name               Sex           Age            SibSp      
 Min.   :  1.0   Min.   :0.0000   Min.   :1.000   Length:891         female:314   Min.   : 0.42   Min.   :0.000  
 1st Qu.:223.5   1st Qu.:0.0000   1st Qu.:2.000   Class :character   male  :577   1st Qu.:22.00   1st Qu.:0.000  
 Median :446.0   Median :0.0000   Median :3.000   Mode  :character                Median :28.86   Median :0.000  
 Mean   :446.0   Mean   :0.3838   Mean   :2.309                                   Mean   :29.59   Mean   :0.523  
 3rd Qu.:668.5   3rd Qu.:1.0000   3rd Qu.:3.000                                   3rd Qu.:36.75   3rd Qu.:1.000  
 Max.   :891.0   Max.   :1.0000   Max.   :3.000                                   Max.   :80.00   Max.   :8.000  
                                                                                                                 
     Parch           Ticket               Fare                Cabin     Embarked   Has_Cabin         Title    
 Min.   :0.0000   Length:891         Min.   :  0.00              :687    :  0    Min.   :0.000   Mr     :517  
 1st Qu.:0.0000   Class :character   1st Qu.:  7.91   B96 B98    :  4   C:168    1st Qu.:0.000   Miss   :182  
 Median :0.0000   Mode  :character   Median : 14.45   C23 C25 C27:  4   Q: 77    Median :0.000   Mrs    :125  
 Mean   :0.3816                      Mean   : 32.20   G6         :  4   S:646    Mean   :0.229   Master : 40  
 3rd Qu.:0.0000                      3rd Qu.: 31.00   C22 C26    :  3            3rd Qu.:0.000   Dr     :  7  
 Max.   :6.0000                      Max.   :512.33   D          :  3            Max.   :1.000   Rev    :  6  
                                                      (Other)    :186                            (Other): 14  
  Family_Size       Surname         
 Min.   : 1.000   Length:891        
 1st Qu.: 1.000   Class :character  
 Median : 1.000   Mode  :character  
 Mean   : 1.905                     
 3rd Qu.: 2.000                     
 Max.   :11.000 
```

##Exploratory Data Analysis
Total Survival Rate
```
> round(prop.table(table(train_data$Survived)),4)

     0      1 
0.6162 0.3838 
```
Sex vs Survival
```
GG <- ggplot(data=train_data)
GG_Avg_Line <- function(aes){GG+geom_bar(aes,position = "fill")+
    geom_hline(yintercept=0.3838, linetype="dashed")+
    geom_text(aes(0,0.3838,label = "Average Survival Rate = 0.3838", vjust = 2,hjust=0),
              family="BL",color="#333333")}

mosaicplot(train_data$Sex ~ train_data$Survived, 
           main="Passenger Fate by Sex", shade=FALSE, 
           color=TRUE, xlab="Sex", ylab="Survived")
```
