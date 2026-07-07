import pickle

model = pickle.load(open("model/model.pkl", "rb"))

hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))

result = model.predict([[hours, attendance]])

print("Predicted Marks:", result[0])