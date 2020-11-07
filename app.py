# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 03:26:03 2019
@author: Utkarsh Kumar
"""

from flask import Flask,render_template,request
from datetime import datetime
import joblib
import pandas as pd
import pytz
import pickle
import numpy as np
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html',template_folder='templates')

@app.route('/predict',methods=['POST'])

def predict():
    #model1 = open('model1_tuned.ml','rb')
# =============================================================================
#     model1 = open('model1_tuned_XG_3_3.ml','rb')
#     model1 = joblib.load(model1)
#     model2 = open('model2_tuned_XG_3_3.ml','rb')
#     model2 = joblib.load(model2)
# =============================================================================

    model1 = open('modelRF1_x','rb')
    model1 = pickle.load(model1)
    model2 = open('modelRF2_x','rb')
    model2 = pickle.load(model2)
        
    #dfmod_dict = open('df_model.ml','rb')
    #dfmod_dict = joblib.load(dfmod_dict)

    dfmod_dict = open("df_modelx.pickle","rb")
    dfmod_dict = pickle.load(dfmod_dict)
    dfmod_dict.set_index(['Model'],inplace=True,drop=True)
    dfmod_dict = dfmod_dict.to_dict()
   
    #dict_details = open('dict.ml','rb')
    #dict_details = joblib.load(dict_details)
    dict_details = open("dictx.pickle","rb")
    dict_details = pickle.load(dict_details)
    #Brand_dict = open('brand.ml','rb')
    #Brand_dict = joblib.load(Brand_dict)
    Brand_dict = open('brandx.pickle','rb')
    Brand_dict = pickle.load(Brand_dict)
    Brand_dict = Brand_dict.sort_values('Brand_encoded')
    Brand_dict.set_index(['Brand_encoded'],inplace =True)
    Brand_dict= Brand_dict.to_dict()
    
    def get_rec1(Rec,Loc):
# =============================================================================
#         df_use = open('df_use.ml','rb')
#         df_fuel = open('df_fuel.ml','rb')
#         df_location = open('df_location.ml','rb')
#         df_model = open('df_model.ml','rb')
#         df_owner = open('df_owner.ml','rb')
#         
#         df_use = joblib.load(df_use)
#         df_fuel = joblib.load(df_fuel)
#         df_location = joblib.load(df_location)
#         df_model = joblib.load(df_model)
#         df_owner = joblib.load(df_owner)
# 
# =============================================================================
        df_use = open('df_usex.pickle','rb')
        df_fuel = open('df_fuelx.pickle','rb')
        df_location = open('df_locationx.pickle','rb')
        df_model = open('df_modelx.pickle','rb')
        df_owner = open('df_ownerx.pickle','rb')
        
        df_use = pickle.load(df_use)
        df_fuel = pickle.load(df_fuel)
        df_location = pickle.load(df_location)
        df_model = pickle.load(df_model)
        df_owner = pickle.load(df_owner)
        
        df_trans= {0:'Automatic',1:'Manual'}
        
        df_fuel.set_index(['Fuel_Type_encoded'],inplace=True,drop=True)
        df_fuel = df_fuel.to_dict()
        df_location.set_index(['Location_encoded'],inplace=True,drop=True)
        df_location = df_location.to_dict()
        df_model.set_index(['Model_encoded'],inplace=True,drop=True)
        df_model = df_model.to_dict()
        df_owner.set_index(['Owner_Type_encoded'],inplace=True,drop=True)
        df_owner = df_owner.to_dict()
                
        
        df_new = df_use[df_use['Location']==Loc]
        df_new['Price_new'] = (Rec-df_new.Price).abs()
        df_new.sort_values(by=['Price_new'],inplace=True)
        df_new.drop(['Price_new'],axis=1,inplace=True)
        rec_arr = df_new.head(20)
        rec_arr.reset_index(drop=True,inplace=True)
        rec_arr.replace({"Model": df_model['Model']},inplace=True)
        rec_arr.replace({"Location": df_location['Location']},inplace=True)
        rec_arr.replace({"Fuel_Type": df_fuel['Fuel_Type']},inplace=True)
        rec_arr.replace({"Transmission": df_trans},inplace=True)
        rec_arr.replace({"Owner_Type": df_owner['Owner_Type']},inplace=True)
        rec_arr_dict = rec_arr.to_dict(orient='records')
        return rec_arr_dict
    
    def datafortableau(df_tableau):
        df_tableau = pd.DataFrame(df_tableau).T
        df_tableau = df_tableau.rename(columns={0:'LOCATION',1:'BRAND',2:'MODEL',3:'YEAR',4:'DISTANCE_COVERED',5:'FUEL_TYPE',6:'TRANSMISSION',7:'OWNER_TYPE',8:'MILEAGE',9:'ENGINE_CAPACITY',10:'POWER',11:'SEATS',12:'PREDICTED_PRICE',13:'DATE'})
        
        df_fuel = open('df_fuel.ml','rb')
        df_location = open('df_location.ml','rb')
        df_model = open('df_model.ml','rb')
        df_owner = open('df_owner.ml','rb')
        df_brand = open('brand.ml','rb')
        
        df_fuel = joblib.load(df_fuel)
        df_location = joblib.load(df_location)
        df_model = joblib.load(df_model)
        df_owner = joblib.load(df_owner)        
        df_brand = joblib.load(df_brand)
        df_trans= {0:'Automatic',1:'Manual'}

        df_fuel.set_index(['Fuel_Type_encoded'],inplace=True,drop=True)
        df_fuel = df_fuel.to_dict()
        df_location.set_index(['Location_encoded'],inplace=True,drop=True)
        df_location = df_location.to_dict()
        df_model.set_index(['Model_encoded'],inplace=True,drop=True)
        df_model = df_model.to_dict()
        df_owner.set_index(['Owner_Type_encoded'],inplace=True,drop=True)
        df_owner = df_owner.to_dict()
        df_brand.set_index(['Brand_encoded'],inplace=True,drop=True)
        df_brand = df_brand.to_dict()
        df_tableau.BRAND =  df_brand.get('Brand').get(df_tableau.BRAND[0])
        df_tableau.FUEL_TYPE = df_fuel.get('Fuel_Type').get(df_tableau.FUEL_TYPE[0])
        df_tableau.LOCATION = df_location.get('Location').get(float(df_tableau.LOCATION[0]))
        #df_tableau.MODEL = df_model.get('Model').get(df_tableau.MODEL[0])
        df_tableau.OWNER_TYPE = df_owner.get('Owner_Type').get(df_tableau.OWNER_TYPE[0])
        df_tableau.TRANSMISSION = df_trans.get(df_tableau.TRANSMISSION[0])
        return df_tableau
        
   
    if request.method == 'POST':
        try:
            val1 = request.form['Loc']
            val_loc = val1
            val8 = request.form['Own']
            val2 = request.form['Brand']
            val2=float(val2)
            val15= Brand_dict.get('Brand').get(val2)
            
            val3 = request.form['Model']
            val4 = request.form['Year']
            val4 = int(val4)
            currentYear = int(datetime.now().year)
            val4 = currentYear-val4
            val5 = request.form['Dist']
            if len(val5) == 0:
                val5=dict_details[val15]['KD']
    
             
            val6 = request.form['Fuel']
            val7 = request.form['Trans']
            if len(val7) == 0:
                val7=dict_details[val15]['Trans']
              
            val9 = request.form['Mil']
            val9=str(val9)
            if len(val9) == 0:
                val9=dict_details[val15]['Mil']
            val9=float(val9)
            val10 = request.form['Cap']
            val9=str(val9)
            if len(val10) == 0:
                val10=dict_details[val15]['Eng']
            val10=float(val10)
            val11 = request.form['Pow']
            val11=str(val11)
            if len(val11) == 0:
                val11=dict_details[val15]['Pow']
            val1=str(val11)
            val12 = request.form['Seats']
            val12=str(val12)
            if len(val12) == 0:
                val12=dict_details[val15]['Set']
                val12=str(val12)
            tz_IN = pytz.timezone('Asia/Calcutta') 
            datetime_IN = datetime.now(tz_IN)
            current_time = datetime_IN.strftime("%d/%m/%Y")
                
            if val3 == "Choose a Model Name":
                pr = "To get more accurate result try giving Model Name  more precise parameters."
                #val3 = dfmod_dict.get("Model_encoded").get(val3)
                val1 = float(val1)
                val2 = float(val2)
                val4 = float(val4)
                val5 = float(val5)
                val6 = float(val6)
                val7 = float(val7)
                val8 = float(val8)
                val9 = float(val9)
                val10 = float(val10)
                val11 = float(val11)
                val12 = float(val12)
# =============================================================================
#                 series = {'Location': val1,'Brand': val2,'Year': val4,'Kilometers_Driven': val5,'Fuel_Type': val6,'Transmission': val7,'Owner_Type':val8,'Mileage': val9,'Engine': val10,'Power':val11,'Seats': val12}
#                 x = pd.DataFrame.from_dict(series,orient='index').T
#                 names = model2.get_booster().feature_names
#                 predict = model2.predict(x[names].iloc[[-1]])
# =============================================================================
        
                data= [val1,val2,val4,val5,val6,val7,val8,val9,val10,val11,val12]
                data = np.array(data)
                data = data.astype(np.float).reshape(1,-1)
                predict = model2.predict(data)
                #valx = val3
                valx='NA'
            else:
                pr = "To get more accurate result try giving more precise parameters."
                valx = val3
                val3 = dfmod_dict.get("Model_encoded").get(val3)
                val1 = float(val1)
                val2 = float(val2)
                val3 = float(val3)
                val4 = float(val4)
                val5 = float(val5)
                val6 = float(val6)
                val7 = float(val7)
                val8 = float(val8)
                val9 = float(val9)
                val10 = float(val10)
                val11 = float(val11)
                val12 = float(val12)
# =============================================================================
#                 series = {'Location': val1,'Brand': val2,'Model': val3,'Year': val4,'Kilometers_Driven': val5,'Fuel_Type': val6,'Transmission': val7,'Owner_Type':val8,'Mileage': val9,'Engine': val10,'Power':val11,'Seats': val12}
#                 x = pd.DataFrame.from_dict(series,orient='index').T
#                 names = model1.get_booster().feature_names
#                 predict = model1.predict(x[names].iloc[[-1]])
# =============================================================================
                
                data= [val1,val2,val3,val4,val5,val6,val7,val8,val9,val10,val11,val12]
                data = np.array(data)
                data = data.astype(np.float).reshape(1,-1)
                predict = model1.predict(data)
    
            #datax= [val_loc,val2,valx,val4,val5,val6,val7,val8,val9,val10,val11,val12,predict[0],current_time]
            #data_tableau = datafortableau(datax) #Returning a dataFrame
            #data_old = pd.read_excel('../Used_car_deploy/Used_car_data.xlsx',mode='w',index=False)
            #data_new = data_tableau.append(data_old,ignore_index=True)
            #data_new.to_excel('../Used_car_deploy/Used_car_data.xlsx',index=False)
            p = predict[0]
            recommend_arr1 = get_rec1(float(p),float(val_loc))     
            return render_template('output.html',template_folder='templates',prediction=predict,rec_arr_dir = recommend_arr1,pr=pr)

        except:
            return render_template('lost.html',template_folder='templates')

@app.route('/recommend',methods=['POST'])
def recommend():
    def get_rec(Rec,Loc):
# =============================================================================
#         df_use = open('df_use.ml','rb')
#         df_fuel = open('df_fuel.ml','rb')
#         df_location = open('df_location.ml','rb')
#         df_model = open('df_model.ml','rb')
#         df_owner = open('df_owner.ml','rb')
#         
#         df_use = joblib.load(df_use)
#         df_fuel = joblib.load(df_fuel)
#         df_location = joblib.load(df_location)
#         df_model = joblib.load(df_model)
#         df_owner = joblib.load(df_owner)        
#         df_trans= {0:'Automatic',1:'Manual'}
#         
# =============================================================================
        df_use = open('df_usex.pickle','rb')
        df_fuel = open('df_fuelx.pickle','rb')
        df_location = open('df_locationx.pickle','rb')
        df_model = open('df_modelx.pickle','rb')
        df_owner = open('df_ownerx.pickle','rb')
        
        df_use = pickle.load(df_use)
        df_fuel = pickle.load(df_fuel)
        df_location = pickle.load(df_location)
        df_model = pickle.load(df_model)
        df_owner = pickle.load(df_owner)
        
        df_trans= {0:'Automatic',1:'Manual'}
        
        df_fuel.set_index(['Fuel_Type_encoded'],inplace=True,drop=True)
        df_fuel = df_fuel.to_dict()
        df_location.set_index(['Location_encoded'],inplace=True,drop=True)
        df_location = df_location.to_dict()
        df_model.set_index(['Model_encoded'],inplace=True,drop=True)
        df_model = df_model.to_dict()
        df_owner.set_index(['Owner_Type_encoded'],inplace=True,drop=True)
        df_owner = df_owner.to_dict()
                
        
        df_new = df_use[df_use['Location']==Loc]
        df_new['Price_new'] = (Rec-df_new.Price).abs()
        df_new.sort_values(by=['Price_new'],inplace=True)
        df_new.drop(['Price_new'],axis=1,inplace=True)
        rec_arr = df_new.head(20)
        rec_arr.reset_index(drop=True,inplace=True)
        rec_arr.replace({"Model": df_model['Model']},inplace=True)
        rec_arr.replace({"Location": df_location['Location']},inplace=True)
        rec_arr.replace({"Fuel_Type": df_fuel['Fuel_Type']},inplace=True)
        rec_arr.replace({"Transmission": df_trans},inplace=True)
        rec_arr.replace({"Owner_Type": df_owner['Owner_Type']},inplace=True)
        rec_arr_dict = rec_arr.to_dict(orient='records')
        return rec_arr_dict
   
    if request.method == 'POST':
        try:
            Rec = request.form['Knpr']
            Rec = float(Rec)
            Rec = Rec/100000
            Loc = request.form['Locc']
            Loc = int(Loc)
            recommend_arr = get_rec(Rec,Loc)
            return render_template('recommend.html',template_folder='templates',rec_arr_dir = recommend_arr)
        except:
            return render_template('lost.html',template_folder='templates')

            
if __name__ == '__main__':
    app.run(debug=True)
