import statistics as st
import pandas as pd
import plotly.express as pe
import plotly.figure_factory as pf
#What exactly is the difference between pf and pg?
import plotly.graph_objects as pg
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("predict.csv")
score = data["TOEFL Score"]
admit=data["Chance of admit"]
#Plotting initial scatter plot with score and chance of admit, and then splitting different parts of the data into two different colors.
scatter1= pe.scatter(x=score,y=admit)
scatter1.show()

#Plotting second scatter plot with two different colors.
colors=[]
for i in admit:
    if(i==0):
        colors.append("red")
    else:
        colors.append("green")
scatter2= pg.Figure(data=pg.Scatter(x=score,y=admit, mode="markers", marker=dict(color=colors)))
scatter2.show()

#Training and testing the AI
scoresframe = data[["GRE Score","TOEFL Score"]]
scorestrain, scorestest, admittrain, admittest, = train_test_split(scoresframe, admit, test_size=0.25, random_state=0)
lr=LogisticRegression()
lr.fit(scorestrain, admittrain)

#Predicting the data, and then seeing how accurate it is.
predict=lr.predict(scorestest)
acscore=accuracy_score(admittest,predict)*100
print("It's accuracy level for the first time:", acscore)

#I would like to repeat the below again and again till it is above 90 with a for loop, but am not sure how to.
if(acscore<90):
    regtestsize=0.10
    print("It seems like the accuracy score wasn't great. Let's try it again, and give the computer more data.")
    scorestrain, scorestest, admittrain, admittest, = train_test_split(scoresframe, admit, test_size=0.10, random_state=0)
    predict2=lr.predict(scorestest)
    acscore2=accuracy_score(admittest,predict2)*100
    print("It's accuracy level again: ", acscore2)


#Finally testing the data with some user input

toefl = int(input("What was your TOEFL score?"))
gre=int(input("What was your GRE score?"))
sc=StandardScaler()
scorestrain = sc.fit_transform(scorestrain)
newresults = sc.transform([[toefl, gre]])
resultuser= lr.predict(newresults)
#Printing the first value the computer gets from the data. If it is 1, then we tell the user they passed, and if it's 0, we tell the user that they didn't.
print(resultuser)
if(resultuser[0]==1):
    print("From my predictions, you will get into the university.")
else:
    print("Hmm...maybe try again next time?")