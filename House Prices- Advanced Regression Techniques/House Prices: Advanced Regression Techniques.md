#House Prices: Advanced Regression Techniques - 1st Stage

##Background
This competition is provided by Kaggle which missioned to predict the house price 

##Main Goals 
1. Mainly focus on data transforming and cleaning
2. Build ML models and sales price, enhance predictivity

##Data Sets
The data stes are provided by Kaggle. The data composes 2919 rows and 80 columns (including sales price, the dependent variable). We separate 1460 of them as tra. Weta and other 1459 of them as testing data. The variable descripions are as below (provided by Kaggle):
```
SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
MSSubClass: The building class
MSZoning: The general zoning classification
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access
Alley: Type of alley access
LotShape: General shape of property
LandContour: Flatness of the property
Utilities: Type of utilities available
LotConfig: Lot configuration
LandSlope: Slope of property
Neighborhood: Physical locations within Ames city limits
Condition1: Proximity to main road or railroad
Condition2: Proximity to main road or railroad (if a second is present)
BldgType: Type of dwelling
HouseStyle: Style of dwelling
OverallQual: Overall material and finish quality
OverallCond: Overall condition rating
YearBuilt: Original construction date
YearRemodAdd: Remodel date
RoofStyle: Type of roof
RoofMatl: Roof material
Exterior1st: Exterior covering on house
Exterior2nd: Exterior covering on house (if more than one material)
MasVnrType: Masonry veneer type
MasVnrArea: Masonry veneer area in square feet
ExterQual: Exterior material quality
ExterCond: Present condition of the material on the exterior
Foundation: Type of foundation
BsmtQual: Height of the basement
BsmtCond: General condition of the basement
BsmtExposure: Walkout or garden level basement walls
BsmtFinType1: Quality of basement finished area
BsmtFinSF1: Type 1 finished square feet
BsmtFinType2: Quality of second finished area (if present)
BsmtFinSF2: Type 2 finished square fee
BsmtUnfSF: Unfinished square feet of basement area
TotalBsmtSF: Total square feet of basement area
Heating: Type of heating
HeatingQC: Heating quality and condition
CentralAir: Central air conditioning
Electrical: Electrical system
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
BsmtFullBath: Basement full bathrooms
BsmtHalfBath: Basement half bathrooms
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
Bedroom: Number of bedrooms above basement level
Kitchen: Number of kitchens
KitchenQual: Kitchen quality
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
Functional: Home functionality rating
Fireplaces: Number of fireplaces
FireplaceQu: Fireplace quality
GarageType: Garage location
GarageYrBlt: Year garage was built
GarageFinish: Interior finish of the garage
GarageCars: Size of garage in car capacity
GarageArea: Size of garage in square feet
GarageQual: Garage quality
GarageCond: Garage condition
PavedDrive: Paved driveway
WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
PoolQC: Pool quality
Fence: Fence quality
MiscFeature: Miscellaneous feature not covered in other categories
MiscVal: $Value of miscellaneous feature
MoSold: Month Sold
YrSold: Year Sold
SaleType: Type of sale
SaleCondition: Condition of sale
```

##Settings
Load in new packages for future usage
```
install.packages('neuralnet') #Neural network
install.packages("e1071") #Neural network
install.packages("gbm") #Gradient Boosting Method
install.packages("xgboost") #Gradient Boosting Method
```

###Getting the Data Into R
```
train <- read.csv("train.csv")
test <- read.csv("test.csv")
sample <- read.csv("sample_submission.csv")
test$SalePrice <- 0 #Build new column
```

###Data structure
str(train)
colnames(train)
summary(train)

```
$ Id           : int  1 2 3 4 5 6 7 8 9 10 ...
 $ MSSubClass   : Factor w/ 16 levels "20","30","40",..: 6 1 6 7 6 5 1 6 5 15 ...
 $ MSZoning     : Factor w/ 5 levels "C (all)","FV",..: 4 4 4 4 4 4 4 4 5 4 ...
 $ LotFrontage  : num  65 80 68 60 84 ...
 $ LotArea      : int  8450 9600 11250 9550 14260 14115 10084 10382 6120 7420 ...
 $ LotShape     : Factor w/ 4 levels "IR1","IR2","IR3",..: 4 4 1 1 1 1 4 1 4 4 ...
 $ LandContour  : Factor w/ 4 levels "Bnk","HLS","Low",..: 4 4 4 4 4 4 4 4 4 4 ...
 $ LotConfig    : Factor w/ 5 levels "Corner","CulDSac",..: 5 3 5 1 3 5 5 1 5 1 ...
 $ LandSlope    : Factor w/ 3 levels "Gtl","Mod","Sev": 1 1 1 1 1 1 1 1 1 1 ...
 $ Neighborhood : Factor w/ 25 levels "Blmngtn","Blueste",..: 6 25 6 7 14 12 21 17 18 4 ...
 $ Condition1   : Factor w/ 9 levels "Artery","Feedr",..: 3 2 3 3 3 3 3 5 1 1 ...
 $ Condition2   : Factor w/ 8 levels "Artery","Feedr",..: 3 3 3 3 3 3 3 3 3 1 ...
 $ BldgType     : Factor w/ 5 levels "1Fam","2fmCon",..: 1 1 1 1 1 1 1 1 1 2 ...
 $ HouseStyle   : Factor w/ 8 levels "1.5Fin","1.5Unf",..: 6 3 6 6 6 1 3 6 1 2 ...
 $ OverallQual  : int  7 6 7 7 8 5 8 7 7 5 ...
 $ OverallCond  : int  5 8 5 5 5 5 5 6 5 6 ...
 $ YearBuilt    : int  2003 1976 2001 1915 2000 1993 2004 1973 1931 1939 ...
 $ YearRemodAdd : int  2003 1976 2002 1970 2000 1995 2005 1973 1950 1950 ...
 $ RoofStyle    : Factor w/ 6 levels "Flat","Gable",..: 2 2 2 2 2 2 2 2 2 2 ...
 $ RoofMatl     : Factor w/ 8 levels "ClyTile","CompShg",..: 2 2 2 2 2 2 2 2 2 2 ...
 $ Exterior1st  : Factor w/ 15 levels "AsbShng","AsphShn",..: 13 9 13 14 13 13 13 7 4 9 ...
 $ Exterior2nd  : Factor w/ 16 levels "AsbShng","AsphShn",..: 14 9 14 16 14 14 14 7 16 9 ...
 $ MasVnrType   : Factor w/ 5 levels "BrkCmn","BrkFace",..: 2 3 2 3 2 3 4 4 3 3 ...
 $ MasVnrArea   : num  196 0 162 0 350 0 186 240 0 0 ...
 $ ExterQual    : Factor w/ 4 levels "Ex","Fa","Gd",..: 3 4 3 4 3 4 3 4 4 4 ...
 $ ExterCond    : Factor w/ 5 levels "Ex","Fa","Gd",..: 5 5 5 5 5 5 5 5 5 5 ...
 $ Foundation   : Factor w/ 6 levels "BrkTil","CBlock",..: 3 2 3 1 3 6 3 2 1 1 ...
 $ BsmtQual     : Factor w/ 5 levels "Ex","Fa","Gd",..: 3 3 3 5 3 3 1 3 5 5 ...
 $ BsmtCond     : Factor w/ 5 levels "Fa","Gd","NA",..: 5 5 5 2 5 5 5 5 5 5 ...
 $ BsmtExposure : Factor w/ 5 levels "Av","Gd","Mn",..: 5 2 3 5 1 5 1 3 5 5 ...
 $ BsmtFinType1 : Factor w/ 7 levels "ALQ","BLQ","GLQ",..: 3 1 3 1 3 3 3 1 7 3 ...
 $ BsmtFinSF1   : int  706 978 486 216 655 732 1369 859 0 851 ...
 $ BsmtFinType2 : Factor w/ 7 levels "ALQ","BLQ","GLQ",..: 7 7 7 7 7 7 7 2 7 7 ...
 $ BsmtFinSF2   : int  0 0 0 0 0 0 0 32 0 0 ...
 $ BsmtUnfSF    : int  150 284 434 540 490 64 317 216 952 140 ...
 $ TotalBsmtSF  : int  856 1262 920 756 1145 796 1686 1107 952 991 ...
 $ Heating      : Factor w/ 6 levels "Floor","GasA",..: 2 2 2 2 2 2 2 2 2 2 ...
 $ HeatingQC    : Factor w/ 5 levels "Ex","Fa","Gd",..: 1 1 1 3 1 1 1 1 3 1 ...
 $ CentralAir   : Factor w/ 2 levels "N","Y": 2 2 2 2 2 2 2 2 2 2 ...
 $ Electrical   : Factor w/ 5 levels "FuseA","FuseF",..: 5 5 5 5 5 5 5 5 2 5 ...
 $ X1stFlrSF    : int  856 1262 920 961 1145 796 1694 1107 1022 1077 ...
 $ X2ndFlrSF    : int  854 0 866 756 1053 566 0 983 752 0 ...
 $ LowQualFinSF : int  0 0 0 0 0 0 0 0 0 0 ...
 $ GrLivArea    : int  1710 1262 1786 1717 2198 1362 1694 2090 1774 1077 ...
 $ BsmtFullBath : int  1 0 1 1 1 1 1 1 0 1 ...
 $ BsmtHalfBath : int  0 1 0 0 0 0 0 0 0 0 ...
 $ FullBath     : int  2 2 2 1 2 1 2 2 2 1 ...
 $ HalfBath     : int  1 0 1 0 1 1 0 1 0 0 ...
 $ BedroomAbvGr : int  3 3 3 3 4 1 3 3 2 2 ...
 $ KitchenAbvGr : int  1 1 1 1 1 1 1 1 2 2 ...
 $ KitchenQual  : Factor w/ 4 levels "Ex","Fa","Gd",..: 3 4 3 3 3 4 3 4 4 4 ...
 $ TotRmsAbvGrd : int  8 6 6 7 9 5 7 7 8 5 ...
 $ Functional   : Factor w/ 7 levels "Maj1","Maj2",..: 7 7 7 7 7 7 7 7 3 7 ...
 $ Fireplaces   : int  0 1 1 1 1 0 1 2 2 2 ...
 $ FireplaceQu  : Factor w/ 6 levels "Ex","Fa","Gd",..: 4 6 6 3 6 4 3 6 6 6 ...
 $ GarageType   : Factor w/ 7 levels "2Types","Attchd",..: 2 2 2 6 2 2 2 2 6 2 ...
 $ GarageYrBlt  : int  90 63 88 85 87 80 91 60 20 28 ...
 $ GarageFinish : Factor w/ 4 levels "Fin","NA","RFn",..: 3 3 3 4 3 4 3 3 4 3 ...
 $ GarageCars   : int  2 2 2 3 3 2 2 2 2 1 ...
 $ GarageArea   : int  548 460 608 642 836 480 636 484 468 205 ...
 $ GarageQual   : Factor w/ 6 levels "Ex","Fa","Gd",..: 6 6 6 6 6 6 6 6 2 3 ...
 $ GarageCond   : Factor w/ 6 levels "Ex","Fa","Gd",..: 6 6 6 6 6 6 6 6 6 6 ...
 $ PavedDrive   : Factor w/ 3 levels "N","P","Y": 3 3 3 3 3 3 3 3 3 3 ...
 $ WoodDeckSF   : int  0 298 0 0 192 40 255 235 90 0 ...
 $ OpenPorchSF  : int  61 0 42 35 84 30 57 204 0 4 ...
 $ EnclosedPorch: int  0 0 0 272 0 0 0 228 205 0 ...
 $ X3SsnPorch   : int  0 0 0 0 0 320 0 0 0 0 ...
 $ ScreenPorch  : int  0 0 0 0 0 0 0 0 0 0 ...
 $ Fence        : Factor w/ 5 levels "GdPrv","GdWo",..: 5 5 5 5 5 3 5 5 5 5 ...
 $ MiscFeature  : Factor w/ 5 levels "Gar2","NA","Othr",..: 2 2 2 2 2 4 2 4 2 2 ...
 $ MiscVal      : int  0 0 0 0 0 700 0 350 0 0 ...
 $ MoSold       : int  2 5 9 2 12 10 8 11 4 1 ...
 $ YrSold       : int  2008 2007 2008 2006 2008 2009 2007 2009 2008 2008 ...
 $ SaleType     : Factor w/ 9 levels "COD","Con","ConLD",..: 9 9 9 9 9 9 9 9 9 9 ...
 $ SaleCondition: Factor w/ 6 levels "Abnorml","AdjLand",..: 5 5 5 1 5 5 5 5 1 5 ...
 $ SalePrice    : int  208500 181500 223500 140000 250000 143000 307000 200000 129900 118000 ...
 $ HavePool     : Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
 ```

##Take a look at the data
```
###Factors
summary(train$MSSubClass) #Relatively concentrate at 20 and 60, well distributed, no NA
train$MSSubClass <- as.factor(train$MSSubClass)
test$MSSubClass <- as.factor(test$MSSubClass)
summary(train$MSZoning) #Concentrate at RL，no NA
summary(train$Street) #6NA, but 1454 Pave
summary(train$Alley) #1369 NAs    TAKE OUT!!!
summary(train$LotShape) #925 Regs, no NA
SalePricePlot(LotShape)
summary(train$LandContour) #1311 Lv1, no NA
summary(train$Utilities) #Only 1 NoSeWa, no NA
summary(train$LotConfig) #1052 Inside, no NA
summary(train$LandSlope) #1382 Gtl, no NA
summary(train$Neighborhood) #Well Distributed, no NA
SalePricePlot(Neighborhood)
summary(train$Condition1) #1260 Norm, no NA
summary(train$Condition2) #1445 Norm, no NA
summary(train$BldgType) #1220 1Fam, no NA
summary(train$HouseStyle) #Well Distributed, no NA
SalePricePlot(HouseStyle)
summary(train$RoofStyle) #1141 Gamle, no NA
summary(train$RoofMatl) #1434 CompShg, no NA
summary(train$Exterior1st) #no NA
SalePricePlot(Exterior1st)
summary(train$Exterior2nd) #no NA
SalePricePlot(Exterior2nd)
summary(train$MasVnrType) #8 NA DONE!!!
summary(train$ExterQual) #no NA
SalePricePlot(ExterQual)
summary(train$ExterCond) #1282 TA, no NA
summary(train$Foundation) #no NA
SalePricePlot(Foundation)
summary(train$BsmtCond) #1311TA, 37 NAs  DONE!!!
summary(train$BsmtExposure) #38 NAs DONE!!!
summary(train$BsmtFinType1) #37 NAs DONE!!!
summary(train$BsmtFinType2) #38 NAs DONE!!!
summary(train$BsmtQual) #37 NAs DONE!!!
summary(train$Heating) #1428 GasA, no NA
summary(train$HeatingQC) #no NA
summary(train$CentralAir) #1365 Y, no NA
summary(train$Electrical) #1334 SBrkr, 1 NA DONE!!! 
summary(train$KitchenQual)#no NA
summary(train$Functional) #no NA
summary(train$FireplaceQu) #690 NAs DONE!!! 
summary(train$GarageType) #81 NAs DONE!!! 
summary(train$GarageFinish) #81 NAs DONE!!! 
summary(train$GarageQual) #81 NAs DONE!!! 
summary(train$GarageCond) #81 NAs DONE!!! 
summary(train$PavedDrive) #no NA 
summary(train$PoolQC) #1453 NAs  DONE!!!
summary(train$Fence) #1179 NAs DONE!!! 
summary(train$MiscFeature) #1406 NAs DONE!!! 
summary(train$SaleType) #1267 WD
summary(train$SaleCondition)
```
###Non-factors
```
attach(train)

SalePricePlot <- function(X)
{ggplot(train)+
    geom_point(aes(x=X,y=SalePrice))}

summary(train$LotFrontage)
SalePricePlot(LotFrontage) #259 NA DONE!!! 

summary(train$LotArea)
SalePricePlot(LotArea) #no NA
SalePricePlot(LotArea)+coord_cartesian(xlim = c(0:100000))
SalePricePlot(LotArea)+coord_cartesian(xlim = c(0:25000))

summary(train$OverallCond)
SalePricePlot(OverallCond) #5 is retty high,no NA

summary(train$YearBuilt)
SalePricePlot(YearBuilt) #Price is higher is it's more recent

summary(train$YearRemodAdd)
SalePricePlot(YearRemodAdd) #Price is higher is it's more recent

summary(train$BsmtFinSF1) #no NA
SalePricePlot(BsmtFinSF1)
ggplot(train)+geom_histogram(aes(x=BsmtFinSF1),binwidth = 3) #Concentrate on 0

summary(train$BsmtUnfSF) #no NA
SalePricePlot(BsmtUnfSF) 
ggplot(train)+geom_histogram(aes(x=BsmtUnfSF),binwidth = 3) #Concentrate on 0

summary(train$BsmtFinSF2) #no NA
SalePricePlot(BsmtFinSF2) 
ggplot(train)+geom_histogram(aes(x=BsmtFinSF2),binwidth = 3) #Concentrate on 0

summary(train$TotalBsmtSF) #no NA
SalePricePlot(TotalBsmtSF) #Bigger basement, higher price
ggplot(train)+geom_histogram(aes(x=TotalBsmtSF),binwidth = 3) 

summary(train$X1stFlrSF) #no NA
SalePricePlot(X1stFlrSF) #The bigger, higher price

summary(train$X2ndFlrSF) #no NA
SalePricePlot(X2ndFlrSF) #The bigger, higher price
ggplot(train)+geom_histogram(aes(x=X2ndFlrSF),binwidth = 3) #Concentrate on 0, most of the houses don;t have 2nd floor

summary(train$LowQualFinSF) #no NA
SalePricePlot(LowQualFinSF) #The bigger, higher price
ggplot(train)+geom_histogram(aes(x=LowQualFinSF),binwidth = 3) #Concentrate on 0

summary(train$GrLivArea) #no NA
SalePricePlot(GrLivArea) #The bigger, higher price

summary(train$FullBath) #no NA
SalePricePlot(FullBath) #The more, higher price

summary(train$GarageYrBlt) #81 NA DONE!!! 
SalePricePlot(GarageYrBlt)

summary(train$PoolArea) #no NA
SalePricePlot(PoolArea) #concentrate at 0, only few have pool

summary(train$MiscVal) #no NA
SalePricePlot(MiscVal) #concentrate at 0, only few have $Value of miscellaneous feature
```

##Data Cleaning
###Creat a code table to summary the structure of data set. 
```
#Code Table
Train_Code_table <- data.frame(Colnum=NA,Col_names=NA,Is_interger=NA,Is_factor=NA)
for(i in c(1:length(train))){
  Train_Code_table[i,1] <- i
  Train_Code_table[i,2] <- colnames(train)[i]
  Train_Code_table[i,3] <- is.integer(train[1,i])
  Train_Code_table[i,4] <- is.factor(train[1,i])
}

Test_Code_table <- data.frame(Colnum=NA,Col_names=NA,Is_interger=NA,Is_factor=NA)
for(i in c(1:length(test))){
  Test_Code_table[i,1] <- i
  Test_Code_table[i,2] <- colnames(test)[i]
  Test_Code_table[i,3] <- is.integer(test[1,i])
  Test_Code_table[i,4] <- is.factor(test[1,i])
}
```

###Correlations of Pools
```
plot(train[,c(31:38)])
table(train$Condition1,train$Condition2)
plot(train[,c(72,73,81)])
```

###Take out pool and combine into one, pool or not
```
table(train$PoolArea,train$PoolQC)
which(train$PoolArea!=0)
which(is.na(train$PoolQC)==F) #Same rows
train$HavePool <- 0
train[train$PoolArea!=0,]$HavePool <- 1 
train$HavePool <- as.factor(train$HavePool)
train$PoolArea <- NULL
train$PoolQC <- NULL

which(test$PoolArea!=0)
which(is.na(test$PoolQC)==F) #Same rows
test$HavePool <- 0
test[c(which(test$PoolArea!=0)),]$HavePool <- 1 
test$PoolArea <- NULL
test$PoolQC <- NULL
```
###Check Correlations of MiscFeature
```
plot(train$SalePrice,train$MiscFeature)
levels(train$MiscFeature)
```

##Model 1: Decission Tree of all variable
```
A <- colnames(train)

M1_DT <- rpart(SalePrice ~ MSSubClass+MSZoning+LotFrontage+LotArea+Street+
                 Alley+LotShape+LandContour+Utilities+LotConfig+LandSlope+
                 Neighborhood+Condition1+Condition2+BldgType+HouseStyle+OverallQual+
                 OverallCond+YearBuilt+YearRemodAdd+RoofStyle+RoofMatl+Exterior1st+
                 Exterior2nd+MasVnrType+MasVnrArea+ExterQual+ExterCond+Foundation+
                 BsmtQual+BsmtCond+BsmtExposure+BsmtFinType1+BsmtFinSF1+BsmtFinType2+
                 BsmtFinSF2+BsmtUnfSF+TotalBsmtSF+Heating+HeatingQC+CentralAir+
                 Electrical+X1stFlrSF+X2ndFlrSF+LowQualFinSF+GrLivArea+BsmtFullBath+
                 BsmtHalfBath+FullBath+HalfBath+BedroomAbvGr+KitchenAbvGr+KitchenQual+
                 TotRmsAbvGrd+Functional+Fireplaces+FireplaceQu+GarageType+GarageYrBlt+
                 GarageFinish+GarageCars+GarageArea+GarageQual+GarageCond+PavedDrive+
                 WoodDeckSF+OpenPorchSF+EnclosedPorch+X3SsnPorch+ScreenPorch+PoolArea+
                 PoolQC+Fence+MiscFeature+MiscVal+MoSold+YrSold+
                 SaleType+SaleCondition,
               data = train,
               method = "anova")
summary(M1_DT)
dev.new(width=10, height=8) #use this to open rpart
fancyRpartPlot(M1_DT)
```
###Predict M1
```
Predict_M1 <- predict(M1_DT, test)
Outcome_M1 <- data.frame(Id=test$Id,SalePrice=Predict_M1) 
colnames(Outcome_M1)
write.csv(Outcome_M1,"Outcome_M1.csv",row.names = F)
```
###Result <- Score=0.2446, Rank=3716/4119

##Model 2: Decision Tree with cleaned data
```
colnames(train)
M2_DT <- rpart(SalePrice~MSSubClass+MSZoning+LotFrontage+
                 LotArea+LotShape+LandContour+Utilities+
                 LotConfig+LandSlope+Neighborhood+Condition1+
                 Condition2+BldgType+HouseStyle+OverallQual+
                 OverallCond+YearBuilt+YearRemodAdd+RoofStyle+
                 RoofMatl+Exterior1st+Exterior2nd+MasVnrType+
                 MasVnrArea+ExterQual+ExterCond+Foundation+
                 BsmtQual+BsmtCond+BsmtExposure+BsmtFinType1+
                 BsmtFinSF1+BsmtFinType2+BsmtFinSF2+BsmtUnfSF+
                 TotalBsmtSF+Heating+HeatingQC+CentralAir+
                 Electrical+X1stFlrSF+X2ndFlrSF+LowQualFinSF+
                 GrLivArea+BsmtFullBath+BsmtHalfBath+FullBath+
                 HalfBath+BedroomAbvGr+KitchenAbvGr+KitchenQual+
                 TotRmsAbvGrd+Functional+Fireplaces+FireplaceQu+
                 GarageType+GarageYrBlt+GarageFinish+GarageCars+
                 GarageArea+GarageQual+GarageCond+PavedDrive+
                 WoodDeckSF+OpenPorchSF+EnclosedPorch+X3SsnPorch+
                 ScreenPorch+Fence+MiscFeature+MiscVal+
                 MoSold+YrSold+SaleType+SaleCondition+
                 HavePool,
               data = train,
               method = "anova")
summary(M2_DT)
dev.new(width=10, height=8) #use this to open rpart
fancyRpartPlot(M2_DT)
```

###Column refine
```
colnames(Outcome_M2)
test$MSSubClass%>%str
levels(test$MSSubClass)
train$MSSubClass%>%str
levels(train$MSSubClass)[16] <- 150

test$MasVnrType%>%levels
levels(train$MasVnrType)[5] <- "NA"

train$GarageYrBlt <- as.integer(train$GarageYrBlt)
levels(train$GarageYrBlt)
test$GarageYrBlt <- as.integer(test$GarageYrBlt)
levels(test$GarageYrBlt)
test$HavePool <- as.factor(test$HavePool)
```
###Predict M2
```
Predict_M2 <- predict(M2_DT,test)
Outcome_M2 <- data.frame(Id=test$Id,SalePrice=Predict_M2) 
write.csv(Outcome_M2,"Outcome_M2.csv",row.names = F)
```

#House Prices: Advanced Regression Techniques - 2nd Stage
###Main goal in the second stage
1. Check that all columns exist both in train and test data sets.
2. Eliminate NAs by predicting methods.
3. Transform original variables into dummy variable. 

##1. Fix train$MasVnrType 
```
##Train
NA_FixMasVnrType <- rpart(MasVnrType ~ MSSubClass+MSZoning+LotFrontage+LotArea+Street+
                 Alley+LotShape+LandContour+Utilities+LotConfig+LandSlope+
                 Neighborhood+Condition1+Condition2+BldgType+HouseStyle+OverallQual+
                 OverallCond+YearBuilt+YearRemodAdd+RoofStyle+RoofMatl+Exterior1st+
                 Exterior2nd+MasVnrArea+ExterQual+ExterCond+Foundation+
                 BsmtQual+BsmtCond+BsmtExposure+BsmtFinType1+BsmtFinSF1+BsmtFinType2+
                 BsmtFinSF2+BsmtUnfSF+TotalBsmtSF+Heating+HeatingQC+CentralAir+
                 Electrical+X1stFlrSF+X2ndFlrSF+LowQualFinSF+GrLivArea+BsmtFullBath+
                 BsmtHalfBath+FullBath+HalfBath+BedroomAbvGr+KitchenAbvGr+KitchenQual+
                 TotRmsAbvGrd+Functional+Fireplaces+FireplaceQu+GarageType+GarageYrBlt+
                 GarageFinish+GarageCars+GarageArea+GarageQual+GarageCond+PavedDrive+
                 WoodDeckSF+OpenPorchSF+EnclosedPorch+X3SsnPorch+ScreenPorch+
                 Fence+MiscFeature+MiscVal+MoSold+YrSold+
                 SaleType+SaleCondition+HavePool,
               data = train,
               method = "class")

NA_FixMasVnrType_Predict <- predict(NA_FixMasVnrType,train)
NA_FixMasVnrType_Predict <- as.data.frame.matrix(NA_FixMasVnrType_Predict)
NA_FixMasVnrType_Predict$MasVnrType <- NA #create a column
MasVnrType <- levels(train$MasVnrType)

NA_FixMasVnrType_Predict[which(NA_FixMasVnrType_Predict[,1]>0.5),]$MasVnrType <- MasVnrType[1]
NA_FixMasVnrType_Predict[which(NA_FixMasVnrType_Predict[,2]>0.5),]$MasVnrType <- MasVnrType[2]
NA_FixMasVnrType_Predict[which(NA_FixMasVnrType_Predict[,3]>0.5),]$MasVnrType <- MasVnrType[3]
NA_FixMasVnrType_Predict[which(NA_FixMasVnrType_Predict[,4]>0.5),]$MasVnrType <- MasVnrType[4]

for (i in c(which(is.na(train$MasVnrType)))){
  train[i,]$MasVnrType <- NA_FixMasVnrType_Predict[i,]$MasVnrType
}

summary(train$MasVnrType)

##Test
Same as Train
```

##2. Fix train$BsmtCond
```
##Train
A <- which(train$BsmtCond =="NA")
which(is.na(train$BsmtExposure)==T)
which(is.na(train$BsmtFinType1)==T)
which(is.na(train$BsmtFinType2)==T)

train_backup$BsmtFinType2

train$BsmtExposure <- as.character(paste(train$BsmtExposure))
train$BsmtExposure <- as.factor(train$BsmtExposure)
train$BsmtFinType1 <- as.character(paste(train$BsmtFinType1))
train$BsmtFinType1 <- as.factor(paste(train$BsmtFinType1))
train$BsmtFinType2 <- as.character(paste(train$BsmtFinType2))
train$BsmtFinType2 <- as.factor(paste(train$BsmtFinType2))
train$BsmtCond <- as.character(paste(train$BsmtCond))
train$BsmtCond <- as.factor(paste(train$BsmtCond))
train$BsmtQual <- as.character(paste(train$BsmtQual))
train$BsmtQual <- as.factor(paste(train$BsmtQual))
train[which(is.na(train$BsmtFinType2)==T),]$BsmtExposure <- "NA"
train[which(is.na(train$BsmtFinType2)==T),]$BsmtFinType1 <- "NA"
train[which(is.na(train$BsmtFinType2)==T),]$BsmtFinType2 <- "NA"
train[which(is.na(train$BsmtFinType2)==T),]$BsmtCond <- "NA"
train[which(train$BsmtQual=="NA"),]$BsmtQual <- "NA"

summary(train$BsmtExposure) #38 NAs
summary(train$BsmtFinType1) #37 NAs
summary(train$BsmtFinType2) #38 NAs
summary(train$BsmtCond)
summary(train$BsmtQual)

##Test
Same as Train
```

##3. Fix Electrical
```
##Train
A <- which(is.na(train$Electrical)==T)
train[A,]$Electrical <- "SBrkr"
summary(train$Electrical)
summary(train)
##Test
Same as Train
```

##4. Fix Fire Places
```
##Train
summary(train$FireplaceQu)
train$FireplaceQu <- as.character(paste(train$FireplaceQu))
train$FireplaceQu <- as.factor(paste(train$FireplaceQu))
summary(train$Fireplaces)
table(train$FireplaceQu,train$Fireplaces) #All NAs have 0 Fireplaces
##Test
Same as Train
```

##5. Fix Garage
```
##Train
summary(test$GarageType)
summary(test$GarageFinish)
summary(test$GarageCond)
summary(test$GarageQual)

A <- which(is.na(train$GarageType)==T)
B <- which(is.na(train$GarageFinish)==T)
C <- which(is.na(train$GarageCond)==T)
D <- which(is.na(train$GarageQual)==T)

A %in% B
A %in% C
A %in% D #NA are all the same

train$GarageType <- as.character(paste(train$GarageType))
train$GarageFinish <- as.character(paste(train$GarageFinish))
train$GarageCond <- as.character(paste(train$GarageCond))
train$GarageQual <- as.character(paste(train$GarageQual))

train$GarageType <- as.factor(train$GarageType)
train$GarageFinish <- as.factor(train$GarageFinish)
train$GarageCond <- as.factor(train$GarageCond)
train$GarageQual <- as.factor(train$GarageQual)

summary(train$GarageCars)
summary(train$GarageYrBlt)
summary(train$GarageArea)

train$GarageYrBlt <- as.character(train$GarageYrBlt)
train[which(is.na(train$GarageYrBlt)==T),]$GarageYrBlt <- "NA"
SalePricePlot(GarageYrBlt)
train$GarageYrBlt <- as.factor(train$GarageYrBlt)

##Test
Same as Train
```

##6. Fix Fence and MiscFeature
```
summary(train$Fence) #1179 NAs
train$Fence <- as.character(paste(train$Fence))
train$Fence <- as.factor(train$Fence)
summary(train$MiscFeature) #1406 NAs
train$MiscFeature <- as.character(paste(train$MiscFeature))
train$MiscFeature <- as.factor(train$MiscFeature)

summary(test$Fence) 
test$Fence <- as.character(paste(test$Fence))
test$Fence <- as.factor(test$Fence)
summary(test$MiscFeature)
test$MiscFeature <- as.character(paste(test$MiscFeature))
test$MiscFeature <- as.factor(test$MiscFeature)
```

##7. Fix LotFrontage
```
##Train
summary(train$LotFrontage)
which(is.na(train$LotFrontage)==T)

NA_LotFrontage <- rpart(LotFrontage ~ MSSubClass+MSZoning+LotArea+Street+
                 Alley+LotShape+LandContour+Utilities+LotConfig+LandSlope+
                 Neighborhood+Condition1+Condition2+BldgType+HouseStyle+OverallQual+
                 OverallCond+YearBuilt+YearRemodAdd+RoofStyle+RoofMatl+Exterior1st+
                 Exterior2nd+MasVnrType+MasVnrArea+ExterQual+ExterCond+Foundation+
                 BsmtQual+BsmtCond+BsmtExposure+BsmtFinType1+BsmtFinSF1+BsmtFinType2+
                 BsmtFinSF2+BsmtUnfSF+TotalBsmtSF+Heating+HeatingQC+CentralAir+
                 Electrical+X1stFlrSF+X2ndFlrSF+LowQualFinSF+GrLivArea+BsmtFullBath+
                 BsmtHalfBath+FullBath+HalfBath+BedroomAbvGr+KitchenAbvGr+KitchenQual+
                 TotRmsAbvGrd+Functional+Fireplaces+FireplaceQu+GarageType+GarageYrBlt+
                 GarageFinish+GarageCars+GarageArea+GarageQual+GarageCond+PavedDrive+
                 WoodDeckSF+OpenPorchSF+EnclosedPorch+X3SsnPorch+ScreenPorch+
                 Fence+MiscFeature+MiscVal+MoSold+YrSold+
                 SaleType+SaleCondition,
                          data = train,
                          method = "anova")

summary(NA_LotFrontage)
fancyRpartPlot(NA_LotFrontage)
dev.new(width=10, height=8)
NA_LotFrontage_Predict <- predict(NA_LotFrontage,train)
NA_LotFrontage_Predict <- as.data.frame(NA_LotFrontage_Predict)
colnames(NA_LotFrontage_Predict)[1] <- "LotFrontage"
RMSE <- sqrt(sum((train[B,"LotFrontage"] - NA_LotFrontage_Predict[B,"LotFrontage"])^2)/nrow(train))

B <- which(is.na(train$LotFrontage)==F)
A <- which(is.na(train$LotFrontage)==T)

train[A,]$LotFrontage <- NA_LotFrontage_Predict[A,"LotFrontage"]
NA_LotFrontage_Predict[A,"LotFrontage"]%>%head
train[A,]$LotFrontage%>%head

##Test
Same as Train
```

##8. Fix MasVnrArea
```
##Train
summary(train$MasVnrType)
summary(train$MasVnrArea)
SalePricePlot(MasVnrArea)
ggplot(train)+
  geom_point(aes(MasVnrType,MasVnrArea))

mean(train$MasVnrArea,na.rm = T)
train[which(is.na(train$MasVnrArea)==T),]$MasVnrArea <- 104
train_backup <- train

##Test
summary(test$MasVnrType)
summary(test$MasVnrArea)
test[which(is.na(test$MasVnrArea)==T),"MasVnrArea"] <- 0
which(is.na(test$MasVnrArea)==T) #1133 1198 is not 0
test[c(1133,1198),"MasVnrArea"] <- round(mean(test$MasVnrArea,na.rm = T),0)

test$MasVnrType <- as.character(paste(test$MasVnrType))
test$MasVnrType <- as.factor(test$MasVnrType)
```

##9. Fix Test SaleType 
```
table(test$SaleType)
test$SaleType <- as.character(test$SaleType)
test[which(is.na(test$SaleType)==T),]$SaleType <- "WD" #WD make up the most
test$SaleType <- as.factor(test$SaleType)
```

##10. Fix Data Errors
```
#Fix Test SaleType 
table(test$SaleType)
test$SaleType <- as.character(test$SaleType)
test[which(is.na(test$SaleType)==T),]$SaleType <- "WD" #WD make up the most
test$SaleType <- as.factor(test$SaleType)

#Fix Test Functional
table(test$Functional)
test$Functional <- as.character(test$Functional)
test[which(is.na(test$Functional)==T),]$Functional <- "Typ" #Typ make up the most
test$Functional <- as.factor(test$Functional)

#Fix Test KitchenQual
table(test$KitchenQual)
test$KitchenQual <- as.character(test$KitchenQual)
test[which(is.na(test$KitchenQual)==T),]$KitchenQual <- "TA" #TA make up the most
test$KitchenQual <- as.factor(test$KitchenQual)

#Fix Bath related columns
which(is.na(test$BsmtHalfBath))
test[661,c("BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF")] <- 0
test[c(661,729),c("BsmtFullBath","BsmtHalfBath")] <- 0


#Fix Exterior1st Exterior2nd
summary(test$Exterior1st)
train$Exterior1st <- as.character(test$Exterior1st)
train$Exterior1st <- as.factor(test$Exterior1st)
test[which(is.na(test$Exterior1st)==T),]$Exterior1st <- "VinylSd"

summary(test$Exterior2nd)
train$Exterior2nd <- as.character(test$Exterior2nd)
train$Exterior1st <- as.factor(test$Exterior2nd)
test[which(is.na(test$Exterior2nd)==T),]$Exterior2nd <- "VinylSd"

# Fix Utilities and MSZoning
which(is.na(test))
test$Utilities %>% summary
test$Utilities <- as.character(test$Utilities)
test[which(is.na(test$Utilities)==T),]$Utilities <- "AllPub"
test$Utilities <- as.factor(test$Utilities)

test$MSZoning %>% summary
test$MSZoning <- as.character(test$MSZoning)
test[which(is.na(test$MSZoning)==T),]$MSZoning <- "RL"
test$MSZoning <- as.factor(test$MSZoning)

#Data Cleaining Complete!!!!!!!!!!!!
```

##Column Transfrom - Create Dummy Variables
```
#reform variable
train_new <- model.matrix(~MSSubClass+MSZoning+LotFrontage+
                            LotArea+LotShape+LandContour+LotConfig+
                            LandSlope+Neighborhood+Condition1+Condition2+
                            BldgType+HouseStyle+OverallQual+OverallCond+
                            YearBuilt+YearRemodAdd+RoofStyle+RoofMatl+
                            Exterior1st+Exterior2nd+MasVnrType+MasVnrArea+
                            ExterQual+ExterCond+Foundation+BsmtQual+
                            BsmtCond+BsmtExposure+BsmtFinType1+BsmtFinSF1+
                            BsmtFinType2+BsmtFinSF2+BsmtUnfSF+TotalBsmtSF+
                            Heating+HeatingQC+CentralAir+Electrical+
                            X1stFlrSF+X2ndFlrSF+LowQualFinSF+GrLivArea+
                            BsmtFullBath+BsmtHalfBath+FullBath+HalfBath+
                            BedroomAbvGr+KitchenAbvGr+KitchenQual+TotRmsAbvGrd+
                            Functional+Fireplaces+FireplaceQu+GarageType+
                            GarageYrBlt+GarageFinish+GarageCars+GarageArea+
                            GarageQual+GarageCond+PavedDrive+WoodDeckSF+
                            OpenPorchSF+EnclosedPorch+X3SsnPorch+ScreenPorch+
                            Fence+MiscFeature+MiscVal+MoSold+
                            YrSold+SaleType+SaleCondition+HavePool,data=train)

train_new <- as.data.frame(train_new)

test_new <- model.matrix(~MSSubClass+MSZoning+LotFrontage+
                           LotArea+LotShape+LandContour+LotConfig+
                           LandSlope+Neighborhood+Condition1+Condition2+
                           BldgType+HouseStyle+OverallQual+OverallCond+
                           YearBuilt+YearRemodAdd+RoofStyle+RoofMatl+
                           Exterior1st+Exterior2nd+MasVnrType+MasVnrArea+
                           ExterQual+ExterCond+Foundation+BsmtQual+
                           BsmtCond+BsmtExposure+BsmtFinType1+BsmtFinSF1+
                           BsmtFinType2+BsmtFinSF2+BsmtUnfSF+TotalBsmtSF+
                           Heating+HeatingQC+CentralAir+Electrical+
                           X1stFlrSF+X2ndFlrSF+LowQualFinSF+GrLivArea+
                           BsmtFullBath+BsmtHalfBath+FullBath+HalfBath+
                           BedroomAbvGr+KitchenAbvGr+KitchenQual+TotRmsAbvGrd+
                           Functional+Fireplaces+FireplaceQu+GarageType+
                           GarageYrBlt+GarageFinish+GarageCars+GarageArea+
                           GarageQual+GarageCond+PavedDrive+WoodDeckSF+
                           OpenPorchSF+EnclosedPorch+X3SsnPorch+ScreenPorch+
                           Fence+MiscFeature+MiscVal+MoSold+
                           YrSold+SaleType+SaleCondition+HavePool,data=test)

test_new <- as.data.frame(test_new)
```

###Look at the difference between train and test
```
setdiff(colnames(train),colnames(test))
setdiff(colnames(train_new),colnames(test_new))
#[1] "Condition2RRAe"     "Condition2RRAn"     "Condition2RRNn"    
#[4] "HouseStyle2.5Fin"   "RoofMatlCompShg"    "RoofMatlMembran"   
#[7] "RoofMatlMetal"      "RoofMatlRoll"       "Exterior1stImStucc"
#[10] "Exterior1stStone"   "Exterior2ndOther"   "HeatingGasA"       
#[13] "HeatingOthW"        "ElectricalMix"      "GarageQualFa"      
#[16] "MiscFeatureTenC" 
```
###Column double checking
```
###Condition2RRAe
train_new$Condition2RRAe
length(which(train_new$Condition2RRAe==1))
test_new$Condition2RRAe
train_new$Condition2RRAe <- NULL
summary(train$Condition2)
summary(test$Condition2)

###Condition2RRAn
train_new$Condition2RRAn
length(which(train_new$Condition2RRAn==1))
test_new$Condition2RRAn
train_new$Condition2RRAn <- NULL
summary(train$Condition2)
summary(test$Condition2)

###Condition2RRNn
train_new$Condition2RRNn
length(which(train_new$Condition2RRNn==1))
test_new$Condition2RRNn
train_new$Condition2RRNn <- NULL
summary(train$Condition2)
summary(test$Condition2)

###HouseStyle2.5Fin
train_new$HouseStyle2.5Fin
length(which(train_new$HouseStyle2.5Fin==1))
test_new$HouseStyle2.5Fin
train_new$HouseStyle2.5Fin <- NULL
summary(train$HouseStyle)
summary(test$HouseStyle)

###Exterior1stImStucc
train_new$Exterior1stImStucc
length(which(train_new$Exterior1stImStucc==1))
test_new$Exterior1stImStucc
train_new$Exterior1stImStucc <- NULL
summary(train$Exterior1st)
summary(test$Exterior1st)
setdiff(levels(train$Exterior1st),levels(test$Exterior1st))

###Exterior1stStone
train_new$Exterior1stStone
length(which(train_new$Exterior1stStone==1))
test_new$Exterior1stStone
train_new$Exterior1stStone <- NULL

###Exterior2ndOther
train_new$Exterior2ndOther
train_new$Exterior2ndOther <- NULL
length(which(train_new$Exterior2ndOther==1))
test_new$Exterior2ndOther
summary(train$Exterior2nd)
summary(test$Exterior2nd)


###HeatingGasA
train_new$HeatingGasA
length(which(train_new$HeatingGasA==1))
test_new$HeatingGasA
train_new$HeatingGasA <- NULL

summary(train$Heating)
summary(test$Heating)

setdiff(levels(train$Heating),levels(test$Heating))

###HeatingOthW
train_new$HeatingOthW
length(which(train_new$HeatingOthW==1))
train_new$HeatingOthW <- NULL
test_new$HeatingOthW

###ElectricalMix
train_new$ElectricalMix
train_new$ElectricalMix <- NULL
length(which(train_new$ElectricalMix==1))
test_new$ElectricalMix
summary(train$Electrical)
summary(test$Electrical)

###MiscFeatureTenC
train_new$MiscFeatureTenC
train_new$MiscFeatureTenC <- NULL
length(which(train_new$MiscFeatureTenC==1))
test_new$MiscFeatureTenC
summary(train$MiscFeature)
summary(test$MiscFeature)

#Checkings!!!
###GarageQualFa !!!
train_new$GarageQualFa
length(which(train_new$GarageQualFa==1))
test_new$GarageQualFa
summary(train$GarageQual)
summary(test$GarageQual)
length(which(train_new$GarageQualEx==1))
length(which(train_new$GarageQualGd==1))
length(which(train_new$GarageQualNA==1))
length(which(train_new$GarageQualPo==1))
length(which(train_new$GarageQualTA==1))
length(which(train_new$GarageQualFa==1))

length(which(test_new$GarageQualEx==1))
length(which(test_new$GarageQualGd==1))
length(which(test_new$GarageQualNA==1))
length(which(test_new$GarageQualPo==1))
length(which(test_new$GarageQualTA==1))
length(which(test_new$GarageQualFa==1))

which(train$GarageQual=="Fa")
which(train_new$GarageQualFa==1)

which(test$GarageQual=="Fa")
test_new$GarageQualFa <- 0
test_new[test$GarageQual=="Fa",]$GarageQualFa <- 1


###RoofMatlCompShg
train_new$RoofMatlCompShg
length(which(train_new$RoofMatlCompShg==1))
test_new$RoofMatlCompShg
summary(train$RoofMatl)
summary(test$RoofMatl)

length(which(train_new$RoofMatlCompShg==1))
length(which(train_new$RoofMatlMembran==1))
train_new$RoofMatlMembran <- NULL
test_new$RoofMatlMembran
length(which(train_new$RoofMatlMetal==1))
train_new$RoofMatlMetal <- NULL
test_new$RoofMatlMetal
length(which(train_new$RoofMatlRoll==1))
train_new$RoofMatlRoll <- NULL
test_new$RoofMatlRoll
length(which(train_new$`RoofMatlTar&Grv`==1))
length(which(train_new$RoofMatlWdShake==1))
length(which(train_new$RoofMatlWdShngl==1))
train_new$RoofMatlCompShg

length(which(test_new$RoofMatlCompShg==1))
length(which(test_new$RoofMatlMembran==1))
length(which(test_new$RoofMatlMetal==1))
length(which(test_new$RoofMatlRoll==1))
length(which(test_new$`RoofMatlTar&Grv`==1))
length(which(test_new$RoofMatlWdShake==1))
length(which(test_new$RoofMatlWdShngl==1))
test_new$RoofMatlCompShg <- 0
length(which(test$RoofMatl=="CompShg"))
which(test$RoofMatl=="CompShg")
test_new[which(test$RoofMatl=="CompShg"),]$RoofMatlCompShg <- 1
setdiff(which(test_new$RoofMatlCompShg==1),which(test$RoofMatl=="CompShg")) #no difference

###Heating
summary(train$Heating)
summary(test$Heating)
length(which(train_new$HeatingGasA==1))
length(which(train_new$HeatingGasW==1))
length(which(train_new$HeatingGrav==1))
length(which(train_new$HeatingWall==1))
length(which(train_new$HeatingOthW==1))
train_new$HeatingOthW <- NULL
length(which(train_new$HeatingFloor==1))

length(which(test_new$HeatingGasA==1))
length(which(test_new$HeatingGasW==1))
length(which(test_new$HeatingGrav==1))
length(which(test_new$HeatingWall==1))
length(which(test_new$HeatingOthW==1))
length(which(train_new$HeatingFloor==1))

test_new$HeatingGasA <- 0
test_new[which(test$Heating=="GasA"),]$HeatingGasA <- 1
```

##Model 3: Random Forest
```
set.seed(1)
train_new_back <- train_new
test_new_back <- test_new
colnames(train_new)[1] <- "Intercept"
colnames(test_new)[1] <- "Intercept"
names(train_new) <- make.names(names(train_new))
names(test_new) <- make.names(names(test_new))

#RoofMatlTar&Grv <- RoofMatlTar.Grv, Exterior1stWd Sdng <- Exterior1stWd.Sdng
#Exterior2ndBrk Cmn <- Exterior2ndBrk.Cmn, Exterior2ndWd Sdng <- Exterior2ndWd.Sdng
#Exterior2ndWd Shng <- Exterior2ndWd.Shng

train_new$RoofMatlTar.Grv
train_new$Exterior1stWd.Sdng
setdiff(names(train_new_back),names(train_new))

train_new$RoofMatlTar

f <- SalePrice~MSSubClass30+MSSubClass40+
  MSSubClass45+MSSubClass50+MSSubClass60+MSSubClass70+
  MSSubClass75+MSSubClass80+MSSubClass85+MSSubClass90+
  MSSubClass120+MSSubClass160+MSSubClass180+
  MSSubClass190+MSZoningFV+MSZoningRH+
  MSZoningRL+MSZoningRM+LotFrontage+LotArea+LotShapeIR2+
  LotShapeIR3+LotShapeReg+LandContourHLS+LandContourLow+
  LandContourLvl+LotConfigCulDSac+LotConfigFR2+
  LotConfigFR3+LotConfigInside+LandSlopeMod+LandSlopeSev+
  NeighborhoodBlueste+NeighborhoodBrDale+
  NeighborhoodBrkSide+NeighborhoodClearCr+
  NeighborhoodCollgCr+NeighborhoodCrawfor+
  NeighborhoodEdwards+NeighborhoodGilbert+
  NeighborhoodIDOTRR+NeighborhoodMeadowV+
  NeighborhoodMitchel+NeighborhoodNAmes+
  NeighborhoodNoRidge+NeighborhoodNPkVill+
  NeighborhoodNridgHt+NeighborhoodNWAmes+
  NeighborhoodOldTown+NeighborhoodSawyer+
  NeighborhoodSawyerW+NeighborhoodSomerst+
  NeighborhoodStoneBr+NeighborhoodSWISU+
  NeighborhoodTimber+NeighborhoodVeenker+
  Condition1Feedr+Condition1Norm+Condition1PosA+
  Condition1PosN+Condition1RRAe+Condition1RRAn+
  Condition1RRNe+Condition1RRNn+Condition2Feedr+
  Condition2Norm+Condition2PosA+Condition2PosN+
  BldgType2fmCon+BldgTypeDuplex+BldgTypeTwnhs+
  BldgTypeTwnhsE+HouseStyle1.5Unf+HouseStyle1Story+
  HouseStyle2.5Unf+HouseStyle2Story+HouseStyleSFoyer+
  HouseStyleSLvl+OverallQual+OverallCond+YearBuilt+
  YearRemodAdd+RoofStyleGable+RoofStyleGambrel+
  RoofStyleHip+RoofStyleMansard+RoofStyleShed+
  RoofMatlCompShg+RoofMatlTar.Grv+RoofMatlWdShake+
  RoofMatlWdShngl+Exterior1stAsphShn+Exterior1stBrkComm+
  Exterior1stBrkFace+Exterior1stCBlock+Exterior1stCemntBd+
  Exterior1stHdBoard+Exterior1stMetalSd+Exterior1stPlywood+
  Exterior1stStucco+Exterior1stVinylSd+Exterior1stWd.Sdng+
  Exterior1stWdShing+Exterior2ndAsphShn+Exterior2ndBrk.Cmn+
  Exterior2ndBrkFace+Exterior2ndCBlock+Exterior2ndCmentBd+
  Exterior2ndHdBoard+Exterior2ndImStucc+Exterior2ndMetalSd+
  Exterior2ndPlywood+Exterior2ndStone+Exterior2ndStucco+
  Exterior2ndVinylSd+Exterior2ndWd.Sdng+Exterior2ndWd.Shng+
  MasVnrTypeBrkFace+MasVnrTypeNone+MasVnrTypeStone+
  MasVnrArea+ExterQualFa+ExterQualGd+
  ExterQualTA+ExterCondFa+ExterCondGd+ExterCondPo+
  ExterCondTA+FoundationCBlock+FoundationPConc+
  FoundationSlab+FoundationStone+FoundationWood+
  BsmtQualFa+BsmtQualGd+BsmtQualNA+BsmtQualTA+
  BsmtCondGd+BsmtCondNA+BsmtCondPo+BsmtCondTA+
  BsmtExposureGd+BsmtExposureMn+BsmtExposureNA+
  BsmtExposureNo+BsmtFinType1BLQ+BsmtFinType1GLQ+
  BsmtFinType1LwQ+BsmtFinType1NA+BsmtFinType1Rec+
  BsmtFinType1Unf+BsmtFinSF1+BsmtFinType2BLQ+
  BsmtFinType2GLQ+BsmtFinType2LwQ+BsmtFinType2NA+
  BsmtFinType2Rec+BsmtFinType2Unf+BsmtFinSF2+
  BsmtUnfSF+TotalBsmtSF+HeatingGasA+HeatingGasW+
  HeatingGrav+HeatingWall+HeatingQCFa+HeatingQCGd+
  HeatingQCPo+HeatingQCTA+CentralAirY+ElectricalFuseF+
  ElectricalFuseP+ElectricalSBrkr+X1stFlrSF+X2ndFlrSF+
  LowQualFinSF+GrLivArea+BsmtFullBath+BsmtHalfBath+
  FullBath+HalfBath+BedroomAbvGr+KitchenAbvGr+
  KitchenQualFa+KitchenQualGd+KitchenQualTA+TotRmsAbvGrd+
  FunctionalMaj2+FunctionalMin1+FunctionalMin2+
  FunctionalMod+FunctionalSev+FunctionalTyp+Fireplaces+
  FireplaceQuFa+FireplaceQuGd+FireplaceQuNA+FireplaceQuPo+
  FireplaceQuTA+GarageTypeAttchd+GarageTypeBasment+
  GarageTypeBuiltIn+GarageTypeCarPort+GarageTypeDetchd+
  GarageTypeNA+GarageYrBlt+GarageFinishNA+GarageFinishRFn+
  GarageFinishUnf+GarageCars+GarageArea+GarageQualFa+
  GarageQualGd+GarageQualNA+GarageQualPo+GarageQualTA+
  GarageCondFa+GarageCondGd+GarageCondNA+GarageCondPo+
  GarageCondTA+PavedDriveP+PavedDriveY+WoodDeckSF+
  OpenPorchSF+EnclosedPorch+X3SsnPorch+ScreenPorch+
  FenceGdWo+FenceMnPrv+FenceMnWw+FenceNA+MiscFeatureNA+
  MiscFeatureOthr+MiscFeatureShed+MiscVal+MoSold+YrSold+
  SaleTypeCon+SaleTypeConLD+SaleTypeConLI+SaleTypeConLw+
  SaleTypeCWD+SaleTypeNew+SaleTypeOth+SaleTypeWD+
  SaleConditionAdjLand+SaleConditionAlloca+
  SaleConditionFamily+SaleConditionNormal+
  SaleConditionPartial+HavePool1

f <- formula(f)

M3_RF <- randomForest(SalePrice~f,
                 data=train_new,
                 importance=TRUE)

summary(M3_RF)
varImpPlot(M3_RF)

A <- sapply(train,class)%>%as.data.frame
B <- sapply(test,class)%>%as.data.frame
cols <- cbind(A,B)
```

##Model 4: SVM
```
##Model4 SVM
M4_SVM <- svm(f,data = train_new)
##Predict M4
Predict_M4 <- predict(M4_SVM,test_new)
Outcome_M4 <- data.frame(Id=test_new$Id,SalePrice=Predict_M4) 
write.csv(Outcome_M4,"Outcome_M4.csv",row.names = F)

##Result <- Score=0.19436, no improvement
```

##Model 5: Gradient Boosted Machine
```
#Model 5 - Gradient Boosted Machine
M5_GBM <- gbm(f,data = train_new, distribution="gaussian", n.trees=300, shrinkage=0.05, interaction.depth=3, n.minobsinnode = 300,
              train.fraction = 1, bag.fraction = 0.5,keep.data=TRUE)

summary(M5_GBM)

##Predict M5
M5_predict <- predict(M5_GBM,test_new, n.trees=300)
Outcome_M5 <- data.frame(Id=test_new$Id,SalePrice=M5_predict)
write.csv(Outcome_M5,"Outcome_M5.csv",row.names = F)

##Result <- Score=0.22506, no improvement
```

