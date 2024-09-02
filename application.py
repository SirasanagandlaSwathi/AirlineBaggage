from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pandas as pd
import pickle
import numpy as np


app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegression.pkl','rb'))
df=pd.read_csv(r"C:/Users/User/OneDrive/Desktop/Airline/archive/baggagecomplaints.csv")
@app.route('/',methods=['GET','POST'])
def index():
    airline=sorted(df['Airline'].unique())
    date=sorted(df['Date'].unique())
    month=sorted(df['Month'].unique())
    year=sorted(df['Year'].unique())
    scheduled=df['Scheduled']
    cancelled=df['Cancelled']
    enplaned=df['Enplaned']
    return render_template('index.html',airline=airline,date=date,month=month,year=year,scheduled=scheduled,cancelled=cancelled,enplaned=enplaned)
@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    Airline=request.form.get('Airline')
    Date=request.form.get('Date')
    Month=request.form.get('Month')
    Year=request.form.get('Year')
    Scheduled=int(request.form.get('Scheduled'))
    Cancelled=int(request.form.get('Cancelled'))
    Enplaned=int(request.form.get('Enplaned'))
    prediction=model.predict(pd.DataFrame(columns=['Airline','Date','Month','Year','Scheduled','Cancelled','Enplaned'],data=np.array([Airline,Date,Month,Year,Scheduled,Cancelled,Enplaned]).reshape(1,7)))
    b=(int(prediction))
    return str(abs(np(b)))
if __name__=='__main__':
    app.run(debug=False)