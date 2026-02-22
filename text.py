from ezml import train_model

model1 = train_model(data="C:\\Users\\ajays\\Downloads\\test\\bad_data.csv", target="hired")

print(model1.predict([[35,100000,10]]))
