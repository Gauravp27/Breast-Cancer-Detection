import numpy as np
import pandas as pd
from flask import Flask, render_template, request, flash
import os
import shutil
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
from flask import *
import mysql.connector
import os

db=mysql.connector.connect(user='root',port=3307,database='breast cancer')
cur=db.cursor()

app = Flask(__name__)

app.config['UPLOAD_FOLDER']=r"uploads"
app.config['SECRET_KEY']='b0b4fbefdc48be27a6123605f02b6b86'

global data, x_train, x_test, y_train, y_test

df = pd.read_csv('DATASET\\data.csv')



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/patient',methods=['POST','GET'])
def patientlog():
    if request.method=='POST':
        email=request.form['Email']
        password=request.form['Password']
        cur.execute("select * from patientreg where Email=%s and Password=%s",(email,password))
        content=cur.fetchall()

        db.commit()
        if content == []:
            msg="Credentials Does't exist"
            return render_template('patientlog.html',msg=msg)
        else:
            msg="Login Successful."
            return render_template('patienthome.html',name=email)
    return render_template('patientlog.html')

@app.route('/patientreg',methods=['POST','GET'])
def patientreg():
    if request.method=='POST':
        name=request.form['Name']
        age=request.form['Age']
        email=request.form['Email']
        password1=request.form['Password']
        password2=request.form['Confirm Password']
        if password1 == password2:
            sql="select * from patientreg where Name='%s' and Email='%s'"%(name,email)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print('----',data)
            if data==[]:
                sql="insert into patientreg(Name,Age,Email,Password) values(%s,%s,%s,%s)"
                val=(name,age,email,password1)
                cur.execute(sql,val)
                db.commit()
                return render_template('patientlog.html')
            else:
                warning='Details already Exist'
                return render_template('patientreg.html',msg=warning)
        error='password not matched'
        flash(error)
    return render_template('patientreg.html')


@app.route('/proceed')
def proceed():
    return render_template('proceed.html')

@app.route('/phome')
def phome():
    return render_template('patienthome.html')




@app.route('/load', methods=["POST","GET"])
def load():
    global df,dataset
    if request.method=="POST":
        file=request.files['file']
        df=pd.read_csv(file)
        dataset=df.head(100)
        msg='DATA LOADED SUCCESSFULLY'
        return render_template('load.html',msg=msg)
    return render_template('load.html')



@app.route('/view')
def view():

    print(dataset)
    print(dataset.head())
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())



@app.route('/preprocess',methods=['POST','GET'])
def preprocess():
    global x, y, size
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        x = df.drop(['diagnosis'], axis=1)
        y = df['diagnosis']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')

    return render_template('preprocess.html')




@app.route('/model', methods= ['GET','POST'])
def model():

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)

    if request.method== 'POST':
        model_no= int(request.form['algo'])

        if model_no==0:
            msg= "You have not selected any model"

        elif model_no == 1:
            model = LogisticRegression()
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            lr = accuracy_score(y_test, pred)*100
            msg = "ACCURACY OF LOGISTIC REGRESSION IS :" + str(lr) + str('%')


        elif model_no== 2:
            cfr = RandomForestClassifier()
            cfr.fit(x_train, y_train)
            pred = cfr.predict(x_test)
            rfcr= accuracy_score(y_test, pred) *100
            msg= "ACCURACY OF RANDOM FOREST CLASSIFIER IS :"+ str(rfcr)+ str('%')




        elif model_no== 3:
            dt = DecisionTreeClassifier()
            dt.fit(x_train, y_train)
            pred = dt.predict(x_test)
            dtac = accuracy_score(y_test, pred)*100
            msg = "ACCURACY OF DECISION TREE CLASSIFIER IS :" + str(dtac)+ str('%')



        elif model_no== 4:
            svm = SVC()
            svm.fit(x_train, y_train)
            pred = svm.predict(x_test)
            accsvm = accuracy_score(y_test, pred)*100
            msg = "ACCURACY OF SUPPORT VECTOR MACHINE IS :" + str(accsvm)+ str('%')


        elif model_no== 5:
            nn=MLPClassifier()
            nn.fit(x_train,y_train)
            pred=nn.predict(x_test)
            nnacc = accuracy_score(y_test, pred)*100

            msg = "ACCURACY OF MULTI-LAYER PERCEPTRON MACHINE IS :" + str(nnacc)+ str('%')


        elif model_no== 6:
            model1 = RandomForestClassifier()
            model2 = LogisticRegression()
            model3 = DecisionTreeClassifier()

            lr = RandomForestClassifier()
            clf_stack = StackingClassifier(classifiers=[model1, model2, model3], meta_classifier=lr, use_probas=True,
                                           use_features_in_secondary=True)
            model_stack = clf_stack.fit(x_train, y_train)
            pred_stack = model_stack.predict(x_test)
            acc_stack = accuracy_score(y_test, pred_stack)*100
            msg = "ACCURACY OF HYBRID MODEL IS :" + str(acc_stack)+ str('%')

        return render_template('model.html', mag = msg)
    return render_template('model.html')


@app.route('/prediction', methods= ['GET', 'POST'])
def prediction():
    global x, y, size, df, x_train,x_test,y_train,y_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)

    if request.method== "POST":
        radius_mean= request.form['radius_mean']
        print(radius_mean)
        texture_mean= request.form['texture_mean']
        print(texture_mean)
        perimeter_mean= request.form['perimeter_mean']
        print(perimeter_mean)
        area_mean= request.form['area_mean']
        print(area_mean)
        smoothness_mean= request.form['smoothness_mean']
        print(smoothness_mean)
        compactness_mean= request.form['compactness_mean']
        print(compactness_mean)
        concavity_mean= request.form['concavity_mean']
        print(concavity_mean)
        concave_points_mean= request.form['concave points_mean']
        print(concave_points_mean)
        symmetry_mean= request.form['symmetry_mean']
        print(symmetry_mean)
        fractal_dimension_mean= request.form['fractal_dimension_mean']
        print(fractal_dimension_mean)
        radius_se = request.form['radius_se']
        print(radius_se)
        texture_se = request.form['texture_se']
        print(texture_se)
        perimeter_se = request.form['perimeter_se']
        print(perimeter_se)
        area_se = request.form['area_se']
        print(area_se)
        smoothness_se = request.form['smoothness_se']
        print(smoothness_se)
        compactness_se = request.form['compactness_se']
        print(compactness_se)
        concavity_se = request.form['concavity_se']
        print(concavity_se)
        concave_points_se = request.form['concave points_se']
        print(concave_points_se)
        symmetry_se = request.form['symmetry_se']
        print(symmetry_se)
        fractal_dimension_se = request.form['fractal_dimension_se']
        print(fractal_dimension_se)
        radius_worst= request.form['radius_worst']
        print(radius_worst)
        texture_worst = request.form['texture_worst']
        print(texture_worst)
        perimeter_worst= request.form['perimeter_worst']
        print(perimeter_worst)
        area_worst = request.form['area_worst']
        print(area_worst)
        smoothness_worst = request.form['smoothness_worst']
        print(smoothness_worst)
        compactness_worst = request.form['compactness_worst']
        print(compactness_worst)
        concavity_worst = request.form['concavity_worst']
        print(concavity_worst)
        concave_points_worst = request.form['concave points_worst']
        print(concave_points_worst)
        symmetry_worst = request.form['symmetry_worst']
        print(symmetry_worst)
        fractal_dimension_worst = request.form['fractal_dimension_worst']
        print(fractal_dimension_worst)



        di= {'radius_mean' : [radius_mean], 'texture_mean' : [texture_mean], 'perimeter_mean' : [perimeter_mean],
             'area_mean' : [area_mean],'smoothness_mean' : [smoothness_mean],'compactness_mean' : [compactness_mean],
             'concavity_mean' : [concavity_mean],'concave points_mean' : [concave_points_mean], 'symmetry_mean' : [symmetry_mean],
             'fractal_dimension_mean' : [fractal_dimension_mean],
             'radius_se' :[radius_se], 'texture_se' :[texture_se], 'perimeter_se':[perimeter_se],'area_se' :[area_se],
             'smoothness_se' :[smoothness_se], 'compactness_se':[compactness_se],'concavity_se' :[concavity_se],
             'concave points_se' :[concave_points_se],'symmetry_se':[symmetry_se],'fractal_dimension_se':[fractal_dimension_se],
             'radius_worst':[radius_worst],'texture_worst':[texture_worst],'perimeter_worst': [perimeter_worst],
             'area_worst': [area_worst],'smoothness_worst' :[smoothness_worst],'compactness_worst' :[compactness_worst],
             'concavity_worst' :[concavity_worst],'concave points_worst' :[concave_points_worst],'symmetry_worst' :[symmetry_worst],
             'fractal_dimension_worst' :[fractal_dimension_worst]}

        test= pd.DataFrame.from_dict(di)
        print(test)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=size, random_state=42)
        nn = MLPClassifier()
        nn.fit(x_train, y_train)
        output = nn.predict(test)
        print(output)

        if output[0] == 'M':
            msg = 'MALIGNANT(These are cancerous)'

        else:
            msg = 'BENIGN(These are not cancerous)'

        return render_template('prediction.html', mag=msg)
    return render_template('prediction.html')





@app.route('/logout')
def logout():
    return redirect(url_for('index'))

if __name__=="__main__":
    app.run(debug=True, port=8000)





