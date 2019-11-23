dataPath = "C:/Users/PK/Downloads/Statfinal"
library(zoo)
AssignmentData<-
  read.csv(file=paste(dataPath,"datastat.csv",sep="/"),
           row.names=1,header=TRUE,sep=",")

head(AssignmentData)
tail(AssignmentData)
str(AssignmentData)
#2 Step 1. Simple regressions
matplot(AssignmentData[,-c(8,9,10)],type='l',ylab="Interest Rates",
        main="History of Interest Rates",xlab="Index")

# mUSGG3M is object of the model with formula USGG3M~Output1 

Output1 = AssignmentData$Output1 # need to assign new name to match with Output1 name in file newPredictor else does not work

mUSGG3M = lm(AssignmentData$USGG3M ~ Output1)
summary(mUSGG3M)
#explore data
c(Total.Variance=var(AssignmentData[,8]),Unexplained.Variance=summary(mUSGG3M)$sigma^2)
coef(mUSGG3M)
matplot(AssignmentData[,1],type="l",col="blue",xaxt="n")
lines(mUSGG3M$fitted.values,col="orange")

newPredictor<-data.frame(Output1=Output1[1])
newPredictor
predict(mUSGG3M,newdata=newPredictor)

### Work
#Fit 7 simple regression models
mUSGG3M = lm(AssignmentData$USGG3M ~ Output1)
mUSGG6M = lm(AssignmentData$USGG6M ~ Output1)
mUSGG2YR = lm(AssignmentData$USGG2YR ~ Output1)
mUSGG3YR = lm(AssignmentData$USGG3YR ~ Output1)
mUSGG5YR = lm(AssignmentData$USGG5YR ~ Output1)
mUSGG10YR = lm(AssignmentData$USGG10YR ~ Output1)
mUSGG30YR = lm(AssignmentData$USGG30YR ~ Output1)

# import newPredictor
newPredictor <- read.table(paste(dataPath,'StatisticalAnalysis_Course_Assignment_1_Data.csv',sep = '/'), header=TRUE)
newPredictor

#Use fitted models for prediction:
#Use function predict() to predict 7 interest rates for the value of newPredictor

prUSGG3M=predict(mUSGG3M,newdata=newPredictor)
prUSGG6M=predict(mUSGG6M,newdata=newPredictor)
prUSGG2YR=predict(mUSGG2YR,newdata=newPredictor)
prUSGG3YR=predict(mUSGG3YR,newdata=newPredictor)
prUSGG5YR=predict(mUSGG5YR,newdata=newPredictor)
prUSGG10YR=predict(mUSGG10YR,newdata=newPredictor)
prUSGG30YR=predict(mUSGG30YR,newdata=newPredictor)

#Save res to a file and upload the file using left sidebar.
predicted.values <- c(`prUSGG3M`, `prUSGG6M`, `prUSGG2YR`, `prUSGG3YR`, `prUSGG5YR`, `prUSGG10YR`, `prUSGG30YR`)
predicted.values
res <- list(predicted.values =  predicted.values)
write.table(res, file = paste(dataPath,'result.csv',sep = '/'), row.names = F)

#Collect all slopes and intercepts in one matrix called simpleRegressionResults and keep this matrix for further use in Step 5.

simpleRegressionResults =data.frame()
s =rbind (mUSGG3M$coefficients,mUSGG6M$coefficients,mUSGG2YR$coefficients,mUSGG3YR$coefficients,mUSGG3YR$coefficients,mUSGG5YR$coefficients,mUSGG10YR$coefficients,mUSGG30YR$coefficients)
simpleRegressionResults = data.frame(s)
#3 Step 2. Logistic regression


AssignmentDataLogistic<-AssignmentData[,c(1:7,10)]
head(AssignmentDataLogistic)
All.NAs<-is.na(AssignmentData[,9])&is.na(AssignmentData[,10]) # neither tightening nor easing
noTightening<-is.na(AssignmentData[,10]) 
AssignmentDataLogistic[noTightening,'Tightening']<-0 # replace NAs with 0

cat("Before: ",dim(AssignmentDataLogistic),"\n") # before

AssignmentDataLogistic<-AssignmentDataLogistic[!All.NAs,]
cat("After: ",dim(AssignmentDataLogistic),"\n") # after removing neutral periods


AssignmentDataLogistic[c(275:284),]

matplot(AssignmentDataLogistic[,-8],type="l",ylab="Data and Binary Fed Mode",ylim=c(-5,20))
lines(AssignmentDataLogistic[,8]*20,col="blue",lwd=3)
legend("topright",legend=c("response"),lty=1,col=c("blue"),lwd=3)

#Estimate logistic regression with 3M yields as predictor and Tightening as output.

LogisticModel_3M<-glm(Tightening~USGG3M,family=binomial(link=logit),AssignmentDataLogistic)
summary(LogisticModel_3M)

#Explore the estimated model LogisticModel_All.

LogisticModel_All = glm(Tightening~USGG3M+USGG6M+USGG2YR+USGG3YR +USGG5YR +USGG10YR +USGG30YR  ,family=binomial(link=logit),AssignmentDataLogistic)
summary(LogisticModel_All)

summary(LogisticModel_All)$aic

summary(LogisticModel_All)$coefficients[,c(1,4)]


matplot(AssignmentDataLogistic[,-8],type="l",ylab="Results of Logistic Regression",ylim=c(-5,20))
lines(AssignmentDataLogistic[,8]*20,col="blue",lwd=3)
lines(LogisticModel_All$fitted.values*20,col="orange",lwd=3)
legend("topright",legend=c("prediction","response"),lty=1,
       col=c("orange","blue"),lwd=3)

# Calculate odds
Log.Odds<-predict(LogisticModel_All)     # predict log-odds
Probabilities<-1/(exp(-Log.Odds)+1)      # predict probabilities

# Plot fitted values and probabilities 
plot(LogisticModel_All$fitted.values,type="l",ylab="Fitted Values & Probabilities",lwd=3,col="blue",main="Fitted Values and Probabilities")
lines(Probabilities,col="orange")
legend("topright",legend=c("Fitted Values","Probabilities"),
       lty=1,lwd=3,col=c("blue","orange"))
#Use logistic regression to predict probabilities of tightening for new input data. Let a vector of 7 numbers be a new predictor vector:
(newPredictors<-unname(unlist(AssignmentDataLogistic[1,1:7])))
newPredictors
#Make it a data frame with the same component names as in the training data:
(newPredictors<-setNames(newPredictors,colnames(AssignmentDataLogistic[1:7])))
(newPredictors<-as.data.frame(t(newPredictors)))
newPredictors
#Use the created data frame as newdata in function predict() with type response.
newPredictors
predict(LogisticModel_All,newdata=newPredictors,type='response')
#Course Assignment. Step 2.
#Read the data from StatisticalAnalysis_Course_Assignment_2_Data.rds.
data2 <- readRDS(paste(dataPath,'StatisticalAnalysis_Course_Assignment_2_Data.rds',sep = '/'))
data2[[1]]
data3 = newPredictors
data3[1:7]=data2[[1]]
data3
predict(LogisticModel_All,newdata=data3,type='response')
summary(LogisticModel_All)

exp(-4.7552*0.01)*100
#Step 3
head(AssignmentDataRegressionComparison)

AssignmentDataRegressionComparison<-AssignmentData[,-c(9,10)]
#complete model 
model_all = lm(Output1~USGG3M+USGG6M+USGG2YR+USGG3YR +USGG5YR +USGG10YR +USGG30YR,AssignmentDataRegressionComparison)
summary(model_all)
#null model
model_null = lm(Output1~1,AssignmentDataRegressionComparison)
summary(model_null)

#Compare models using '''anova()'''
anova(model_all,model_null)
c(Full=AIC(model_all),Null=AIC(model_null))

#Given US Treasury rate = USGG3M
#Let the given rate be testRate. Let additionalRate be one of the remaining rates

mUSGG3M.2 = lm(Output1 ~ USGG3M , AssignmentData)
mUSGG6M.2 = lm(Output1 ~ USGG3M+USGG6M , AssignmentData)
mUSGG2YR.2 = lm(Output1 ~ USGG3M+USGG2YR , AssignmentData)
mUSGG3YR.2 = lm(Output1 ~ USGG3M+USGG3YR , AssignmentData)
mUSGG5YR.2 = lm(Output1 ~ USGG3M+USGG5YR , AssignmentData)
mUSGG10YR.2 = lm(Output1 ~ USGG3M+USGG10YR , AssignmentData)
mUSGG30YR.2 = lm(Output1 ~ USGG3M+USGG30YR , AssignmentData)

summary(mUSGG3M.2)

SumSq = c(anova(mUSGG3M.2,mUSGG6M.2)$'Sum of Sq'[2],
anova(mUSGG3M.2,mUSGG2YR.2)$'Sum of Sq'[2],
anova(mUSGG3M.2,mUSGG3YR.2)$'Sum of Sq'[2],
anova(mUSGG3M.2,mUSGG5YR.2)$'Sum of Sq'[2],
anova(mUSGG3M.2,mUSGG10YR.2)$'Sum of Sq'[2],
anova(mUSGG3M.2,mUSGG30YR.2)$'Sum of Sq'[2])

Selected.US.Treasury = 'USGG5YR'

AIC = c(
AIC(mUSGG6M.2),
AIC(mUSGG2YR.2),
AIC(mUSGG3YR.2),
AIC(mUSGG5YR.2),
AIC(mUSGG10YR.2),
AIC(mUSGG30YR.2))
AIC


res <- list(SumSq= SumSq,AIC = AIC,Selected.US.Treasury = Selected.US.Treasury)
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))

#done with step 3
# Step 4. Rolling window analysis
# Set window width and window shift parameters for rolling window.

Window.width<-20; Window.shift<-5
# Means
all.means<-rollapply(AssignmentDataRegressionComparison,width=Window.width,
                     by=Window.shift,by.column=TRUE, mean)
head(all.means)

Count<-1:dim(AssignmentDataRegressionComparison)[1]
Rolling.window.matrix<-rollapply(Count,width=Window.width,by=Window.shift,by.column=FALSE,FUN=function(z) z)
Rolling.window.matrix[1:10,]    # sequence of rolling windows

Points.of.calculation<-Rolling.window.matrix[,10]    

Means.forPlot<-rep(NA,dim(AssignmentDataRegressionComparison)[1])
Means.forPlot[Points.of.calculation]<-all.means[,1]
cbind(1:25,Means.forPlot[1:25])


cbind(originalData=AssignmentDataRegressionComparison[,1],rollingMeans=Means.forPlot)[1:25,]

plot(AssignmentDataRegressionComparison[,1],type="l",col="blue",lwd=2,
     ylab="Interest Rate & Rolling Mean", main="Rolling Mean of USGG3M")
points(Means.forPlot,col="orange",pch=1)
legend("topright",legend=c("USGG3M","Rolling Mean"),col=c("blue","orange"),lwd=2)

#(Skipped Code)
rolling.dates = AssignmentDataRegressionComparison[1,]
for (i in 2:nrow(AssignmentDataRegressionComparison)) {
  result = AssignmentDataRegressionComparison[i,] - AssignmentDataRegressionComparison[i-1,]
  rolling.dates = rbind(rolling.dates,result)
}
rolling.dates = rolling.dates[-1,]
head(rolling.dates)
#head(round(rolling.sd,3))
rolling.dates<-rollapply(AssignmentDataRegressionComparison[-1,],
                         width=Window.width,by=Window.shift,
                         by.column=FALSE,FUN=function(z) rownames(z))

# Rolling lm coefficients
Coefficients<-rollapply(AssignmentDataRegressionComparison,
                        width=Window.width,
                        by=Window.shift,by.column=FALSE,
                        FUN=function(z) coef(lm(Output1~USGG3M+USGG5YR+USGG30YR,
                                                data=as.data.frame(z))))

rolling.dates<-rollapply(AssignmentDataRegressionComparison[,1:8],
                         width=Window.width,by=Window.shift,by.column=FALSE,
                         FUN=function(z) rownames(z))

rownames(Coefficients)<-rolling.dates[,10]
head(Coefficients)

idxDate<-match("9/22/2005",rolling.dates[,10])
cfnts = c(Coefficients[idxDate,])
cfnts
##Find R-squared values of rolling-window-fit linear models:
# R-squared
r.squared<-rollapply(AssignmentDataRegressionComparison,
                     width=Window.width,by=Window.shift,by.column=FALSE,
                     FUN=function(z) summary(lm(Output1~USGG3M+USGG5YR+USGG30YR,
                                                data=as.data.frame(z)))$r.squared)
r.squared<-data.frame(Date=rolling.dates[,10],R2=r.squared)

head(r.squared)
testDate="9/22/2005"
idxDate<-match("9/22/2005",rolling.dates[,10])
rsqrd = c(r.squared[idxDate,2])

rsqrd
#Analyze the rolling p-values of slopes. What were dates of highest p-values (most insignificant slopes)?
# P-values
# P-values
Pvalues<-rollapply(AssignmentDataRegressionComparison,width=Window.width,
                   by=Window.shift,by.column=FALSE,
                   FUN=function(z) summary(lm(Output1~USGG3M+USGG5YR+USGG30YR,                                                                              data=as.data.frame(z)))$coefficients[,4])
rownames(Pvalues)<-rolling.dates[,10]
Pvalues[1:6,]

pvls=c(Pvalues[idxDate,])
pvls
#find fitted value
#test fitted
fitted<-rollapply(AssignmentDataRegressionComparison,
                     width=Window.width,by=Window.shift,by.column=FALSE,
                     FUN=function(z) fitted(lm(Output1~USGG3M+USGG5YR+USGG30YR,
                                                data=as.data.frame(z))))
fitted
rownames(fitted) = rolling.dates[,10]
head(fitted)

prdctn = (c(fitted[idxDate,10]))
prdctn

#predictor with highest coefficients
highestSensitivity = 'USGG5YR'

#result
res<-list(Date=testDate,Coefficients=cfnts,
          P_values=pvls,R_squared=rsqrd,Prediction=prdctn,
          HighSensitivity=highestSensitivity)
res
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))
# end of step 4
# step 5 PCA
# Perform PCA with the inputs (columns 1-7).
AssignmentDataPCA<-AssignmentData[,1:7]
dim(AssignmentDataPCA)
#Explore the dimensionality of the set of 3M, 2Y and 5Y yields.
# Select 3 variables. Explore dimensionality and correlation 
AssignmentData.3M_2Y_5Y<-AssignmentDataPCA[,c(1,3,5)]
ggpairs(AssignmentData.3M_2Y_5Y)
library(rgl);rgl.points(AssignmentData.3M_2Y_5Y)

#load course data
data <- readRDS(paste(dataPath,'StatisticalAnalysis_Course_Assignment_5_Data.rds',sep = '/'))
data
Maturities<-c(.25,.5,2,3,5,10,30)
#See importance of factors.
Eigen.Decomposition = princomp(AssignmentDataPCA)
Eigen.Decomposition

#find loadings
Loadings = Eigen.Decomposition$loadings
#find factors 
Factors = Eigen.Decomposition$scores

Loadings[,1]<--Loadings[,1]
Factors[,1]<--Factors[,1]

matplot(Factors,type="l",col=c("black","blue","orange"),lty=1,lwd=3,main="Factors After Sign Change")
legend("topright",legend=c("F1","F2","F3"),col=c("black","blue","orange"),lty=1)

matplot(Maturities,Loadings,type="l",lty=1,col=c("black","blue","orange"),lwd=3,main="Loadings After Sign Change")
legend("bottomright",legend=c("L1","L2","L3"),col=c("black","blue","orange"),lty=1)


#Analyze the adjustments that each factor makes to the term curve.
OldCurve<-AssignmentDataPCA[135,]
NewCurve<-AssignmentDataPCA[136,]
CurveChange<-NewCurve-OldCurve
FactorsChange<-Factors[136,]-Factors[135,]
ModelCurveAdjustment.1Factor<-OldCurve+t(Loadings[,1])*FactorsChange[1]
ModelCurveAdjustment.2Factors<-OldCurve+t(Loadings[,1])*FactorsChange[1]+t(Loadings[,2])*FactorsChange[2]
ModelCurveAdjustment.3Factors<-OldCurve+t(Loadings[,1])*FactorsChange[1]+t(Loadings[,2])*FactorsChange[2]+
  t(Loadings[,3])*FactorsChange[3]
matplot(Maturities,
        t(rbind(OldCurve,NewCurve,ModelCurveAdjustment.1Factor,ModelCurveAdjustment.2Factors,
                ModelCurveAdjustment.3Factors)),
        type="l",lty=c(1,1,2,2,2),col=c("black","red","green","blue","magenta"),lwd=3,ylab="Curve Adjustment")
legend(x="topright",c("Old Curve","New Curve","1-Factor Adj.","2-Factor Adj.",
                      "3-Factor Adj."),lty=c(1,1,2,2,2),lwd=3,col=c("black","red","green","blue","magenta"))

#start step 5
#Step5 data Using 7 interest rates in RegressionAssignmentData2014.csv fit PCA model by princomp() with 3 factors.
head(data)
#Data for PCA
head(AssignmentDataPCA)
Ass.pca = princomp(AssignmentDataPCA)

#get loadings.
Ass.pca.loading<-Ass.pca$loadings
#get factors
Ass.pca.factors<-Ass.pca$scores
#Create new data frame with principal components as predictors.
Ass.pca.for.fit<-as.data.frame(cbind(Output=AssignmentData[,8],Ass.pca.factors))

#model
mod = lm(Output ~ Comp.1 + Comp.2 + Comp.3 , data = Ass.pca.for.fit)
summary(mod)
#test data
testdate= "8/17/2012"
#1. Find 3 PCA factors corresponding to testDate
factValue = Ass.pca.factors[testdate,1:3]
#2.Find values of the 3 PCA loadings for testmaturity
loadValue = Ass.pca.loading["USGG6M",1:3]
#3.Predict value of time series
Ass.pca.for.fit2<-as.data.frame(cbind(USGG6M=AssignmentData[,2],Ass.pca.factors))
mod3 = lm(USGG6M ~ Comp.1 + Comp.2 + Comp.3, data = Ass.pca.for.fit2)
summary(mod3)
tmp = mod3$fitted.values
predictValue= tmp[testdate]
#4.Calculate residual of the model on testDate
tmp2 = mod3$residuals
residValue = tmp2[testdate]
#5.Calculate *changes* in all 7 rates of the term curve corresponding to the given increments of the 3 factors
FactorsChange = data$factorsChanges ;FactorsChange

curveChanges =t(Ass.pca.loading[,1])*FactorsChange[1]+
  t(Ass.pca.loading[,2])*FactorsChange[2]+
  t(Ass.pca.loading[,3])*FactorsChange[3]

curveChanges = unlist(curveChanges)

#Create list variable res
res<-list(factValue=factValue,
          loadValue=loadValue,
          predictValue=predictValue,
          residValue=residValue,
          curveChanges=curveChanges)
res # the curvechanges still wrong
saveRDS(res, file = paste(dataPath,'result.rds',sep = '/'))


