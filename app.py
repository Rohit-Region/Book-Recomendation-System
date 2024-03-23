#from distutils.command.upload import upload
from http import client
import string
import webbrowser
#from pickle import GET
#from tabnanny import check
from scipy.sparse import coo_matrix, hstack
from unicodedata import category, name
from sklearn.neighbors import NearestNeighbors
from importlib_metadata import method_cache
import pandas as pd
from flask import Flask , render_template , request,url_for,redirect
from gevent import config
from numpy import append, float_power, int0
# from flask_pymongo import PyMongo
from pymongo import MongoClient
import urllib.request
#from PIL import Image
import os
import numpy as np
import scipy.stats
import seaborn as sns
import test
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from requests import head
from sqlalchemy import false, null
from sympy import O, list2numpy


# import pyrebase
# import dns

People_folder=os.path.join('template','assets','images')
app = Flask(__name__,static_url_path="",static_folder="templates/assets/images")
app.config['uploadfolder']=People_folder



client = MongoClient("mongodb://localhost:27017/")

db=client["Book_Sample"]
col=db["full_book"]
all_records = col.find()
list_cursor=[]
for row in all_records:
    list_cursor.append(row)


#print(test())

all_records = col.find()
list_cursor=[]
for row in all_records:
    list_cursor.append(row)

heading=['IMAGE','TITLE','AUTHOR','CATEGORY','RATING','RATE']

reader=pd.DataFrame.from_records(list_cursor)    
reader1=pd.DataFrame.from_records(list_cursor)
#print("Reader 1 ",reader1)

reader2=reader1.drop(['_id','id','index'],axis=1)
#print("Reader 3 ",reader2)

reader4=reader1.drop(['_id'],axis=1)

book_cf=reader1.drop(['_id','rating'],axis=1)
print(book_cf)

book_cf_1=reader1.drop(['_id'],axis=1)
print(book_cf_1)

client = MongoClient("mongodb://localhost:27017/")

db=client["Book_Sample"]
records = db.todos
col=db["user_cf1"]
all_records = col.find()
list_cursor=[]
for row in all_records:
    list_cursor.append(row)

cf1=pd.DataFrame.from_records(list_cursor)
cf1=cf1.drop(['_id'],axis=1)
print(cf1)

ratings=pd.DataFrame.from_records(list_cursor)
ratings=ratings.drop(['_id'],axis=1)
print(ratings)


new1=reader2.sort_values(by=['rating'],ascending=False)
new1=new1.drop(['category id'],axis=1)
new1=new1.head(20)
new2=new1.values.tolist()
#print("DATA 1 ",new2)
list1=list(new1.columns)
#print("List 1 ",list1)



@app.route("/", methods = ['GET','POST'])
def hello():  
    fullfilename=os.path.join(app.config['uploadfolder'],'book.gif')
    return render_template("index.html",userimage=fullfilename,n1=new2,lis=heading)
    

@app.route("/sign_in", methods = ['GET','POST'])
def sign_in():
    message = 'Please login to your account'
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        email_found = records.find_one({"email": email})            
        password_found = records.find_one({"password": password})
        if email_found:
            if password_found:
                message = 'This email already exists in database'
                return render_template('logged_in.html',email=email,message=message) 
            else:
                return render_template('sign_in.html',message=message)
        else:
            message = 'Email not found'            
            return render_template('sign_in.html', email=email)

    return render_template('sign_in.html', message=message)
 

@app.route("/after_login", methods = ['GET','POST'])
def after_login():
    ratings=pd.DataFrame.from_records(list_cursor)
    ratings=ratings.drop(['_id'],axis=1)
    #print(ratings)
    counts1 = ratings['userid'].value_counts()
    #print(counts1)
    ratings = ratings[ratings['userid'].isin(counts1[counts1 >=200].index)]
    counts = ratings['rating'].value_counts()
    #print(counts)
    ratings = ratings[ratings['rating'].isin(counts[counts >=100].index)]

    combine_book_rating = pd.merge(ratings, book_cf, on='id')
    columns = ['category','author','category id','imageurl','index']
    combine_book_rating = combine_book_rating.drop(columns, axis=1)
    #print(combine_book_rating)


    combine_book_rating = combine_book_rating.dropna(axis = 0, subset = ['title'])

    book_ratingCount = (combine_book_rating.groupby(by = ['title'])['rating'].count().reset_index().rename(columns = {'rating': 'totalRatingCount'})[['title', 'totalRatingCount']])
    #print(book_ratingCount.head())


    rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'title', right_on = 'title', how = 'left')
    #print(rating_with_totalRatingCount.head())

    popularity_threshold = 1
    rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
    #print(rating_popular_book.head())

    combined = rating_popular_book.merge(ratings, left_on = 'userid', right_on = 'userid', how = 'left')
    #print(combined)


    combined = combined.drop_duplicates(['userid', 'title'])
    #print(combined)
    combined_1 = combined.pivot(index = 'title', columns = 'userid', values = 'rating_x').fillna(0)
    combined_1=combined_1.astype(float)
    combined_rating_matrix_1 = csr_matrix(combined_1)
    combined_rating_matrix=combined_rating_matrix_1.toarray()
    #print(combined_rating_matrix)
    

    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(combined_rating_matrix)
    query_index = np.random.choice(combined_1.shape[0])
    distances, indices = model_knn.kneighbors(combined_1.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
    combined_1.index[query_index]
    book=[]
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Recommendations for {0}:\n'.format(combined_1.index[query_index]))
        else:
            book.append(combined_1.index[indices.flatten()[i]])

    #print(book)
    book_1=list(book_cf_1['title'])
    # print(book_1)
    x_1=[]
    for i in range(len(book_1)):
        for j in range(len(book)):
            if book_1[i] == book[j]:
                x_1.append(i)
    print(x_1)

    df1=book_cf_1.iloc[x_1]
    print(df1)

    df1=df1.drop(['id','category id','index'],axis=1)
    df_1=df1.sort_values('rating',ascending=False)
    df2=df_1.values.tolist()
    print("Print the Book name:")
    print(df_1.title)


    return render_template("after_login.html",n2=df2,lis1=heading)

    

@app.route("/rating", methods = ['GET','POST'])
def rating():
    return render_template("rating.html")

@app.route("/contactus", methods = ['GET','POST'])
def contactus():
    return render_template("contactus.html")

@app.route("/aboutus", methods = ['GET','POST'])
def aboutus():
    return render_template("aboutus.html")


@app.route("/login__create", methods = ['GET','POST'])
def login__create():
    if request.method == "POST":

        user = request.form.get("username")
        email = request.form.get("email")        
        password = request.form.get("password")
        user_found = records.find_one({"name": user})
        email_found = records.find_one({"email": email})
        print(user)
        print(email)
        print(password)

        if user_found:
            message = 'There already is a user by that name'
            return render_template('login__create.html', message=message)

        if email_found:
            message = 'This email already exists in database'
            return render_template('login__create.html', message=message)
        else:
            user_input = {'name': user, 'email': email, 'password': password}
            records.insert_one(user_input)
            
            user_data = records.find_one({"email": email})
            new_email = user_data['email']

        db.todos.insert_one({'username': user,'Email': email,'password': password})

        return render_template('logged_in.html',email=email)

    return render_template("login__create.html")


@app.route("/collection", methods = ['GET','POST'])
def collection():
    return render_template("collection.html")

@app.route("/sub1" , methods = ['GET','POST'])
def submit1():
    if request.method == 'POST':
        reader2=reader1.drop(['_id','id','index'],axis=1)
        if request.form.get('action1') == '0':
            reader2 = reader2.loc[reader2['category id'] == '0']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action2') == '1':
            reader2 = reader2.loc[reader2['category id'] == '1']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action3') == '2':
            reader2 = reader2.loc[reader2['category id'] == '2']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action4') == '3':
            reader2 = reader2.loc[reader2['category id'] == '3']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action6') == '5':
            reader2 = reader2.loc[reader2['category id'] == '5']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action7') == '6':
            reader2 = reader2.loc[reader2['category id'] == '6']   
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action8') == '7':
            reader2 = reader2.loc[reader2['category id'] == '7']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action9') == '8':
            reader2 = reader2.loc[reader2['category id'] == '8']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action10') == '9':
            reader2 = reader2.loc[reader2['category id'] == '9']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action11') == '10':
            reader2 = reader2.loc[reader2['category id'] == '10']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action12') == '11':
            reader2 = reader2.loc[reader2['category id'] == '11']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action13') == '12':
            reader2 = reader2.loc[reader2['category id'] == '12']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action14') == '13':
            reader2 = reader2.loc[reader2['category id'] == '13']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action15') == '14':
            reader2 = reader2.loc[reader2['category id'] == '14']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action16') == '15':
            reader2 = reader2.loc[reader2['category id'] == '15']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action17') == '16':
            reader2 = reader2.loc[reader2['category id'] == '16']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action18') == '17':
            reader2 = reader2.loc[reader2['category id'] == '17']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action19') == '18':
            reader2 = reader2.loc[reader2['category id'] == '18']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action20') == '19':
            reader2 = reader2.loc[reader2['category id'] == '19']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action21') == '20':
            reader2 = reader2.loc[reader2['category id'] == '20']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action22') == '21':
            reader2 = reader2.loc[reader2['category id'] == '21']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action23') == '22':
            reader2 = reader2.loc[reader2['category id'] == '22']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action24') == '23':
            reader2 = reader2.loc[reader2['category id'] == '23']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action25') == '24':
            reader2 = reader2.loc[reader2['category id'] == '24']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action26') == '25':
            reader2 = reader2.loc[reader2['category id'] == '25']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action27') == '26':
            reader2 = reader2.loc[reader2['category id'] == '26']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action28') == '27':
            reader2 = reader2.loc[reader2['category id'] == '27']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action29') == '28':
            reader2 = reader2.loc[reader2['category id'] == '28']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action30') == '29':
            reader2 = reader2.loc[reader2['category id'] == '29']
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action31') == '30':
            reader2 = reader2.loc[reader2['category id'] == '30'] 
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action32') == '31':
            reader2 = reader2.loc[reader2['category id'] == '31'] 
            reader2 = reader2.drop(['category id'],axis=1)

        elif request.form.get('action5') == '4':
            reader2 = reader2.loc[reader2['category id'] == '4']
            reader2 = reader2.drop(['category id'],axis=1)

        
        
        category_1=reader2.values.tolist()
        #print("DATA 1 ",category_1)


    return render_template("sub1.html",lis=heading,dat=category_1)


@app.route("/sub" , methods = ['GET','POST'])
def submit():
    if request.method == "POST":
        name = request.form["location"]
        people = request.form['people']
        
        #print(name)
        #print(people)
        
        reader3=reader1.drop(['_id','index'],axis=1)
        if people== '32':
            reader3['book_search']=reader3['title'].str.contains((name),case=False,regex=False)
            data2=reader3[reader3['book_search']==True]
            data2=data2.drop(['book_search'],axis=1)
            data3=data2.drop(['id','category id'],axis=1)
           # data3=data3.sort_values('rating',ascending=False)
            data3=data3.head(15)
            #print(len(data2))
        else:
            reader3 = reader3.loc[reader2['category id'] == people]
            reader3['book_search']=reader3['title'].str.contains((name),case=False,regex=False)
            data2=reader3[reader3['book_search']==True]
            data2=data2.drop(['book_search'],axis=1)
            data3=data2.drop(['id','category id'],axis=1)
           # data3=data3.sort_values('rating',ascending=False)
            data3=data3.head(15)
           # print(len(data2))

        data_2=data3.values.tolist()
        #print("DATA 1 ",data_2)

    data2['author']= data2['author'].astype(str)
    features=data2[['title','category','author']].T.agg(','.join)
    data2['combined_feature']=features
    
    try:
        cm=CountVectorizer().fit_transform(data2['combined_feature'])
        print(cm)
        cs_1=cm.toarray()
        print(cs_1)
        cs=cosine_similarity(cs_1)
        print(cs)

        scores = list(enumerate(cs[0]))
        #print(scores)

        sorted_scores=sorted(scores,key = lambda x:x[1],reverse=True)
        sorted_scores=sorted_scores[1:]
        print(sorted_scores)

        list_1=[]
        for i in range(0,len(sorted_scores)):
            list_1.append(sorted_scores[i][0])
        #print(list_1)

        df1 = data2.loc[data2.index[list_1]]
        #print(df1.head(5))
        
        df1=df1.drop(['id','combined_feature','category id'],axis=1)
        df_1=df1.sort_values('rating',ascending=False)
        df1=df_1.head(10)
        df2=df1.values.tolist()
        print("Book name ",df1.title)

    except ValueError:
        return render_template("sub.html",n=name,m=people,n1=data_2,lis=heading,lis1=heading)
    
   
    return render_template("sub.html",n=name,m=people,n1=data_2,lis=heading,n2=df2,lis1=heading)

if __name__ == "__main__":
    app.run(debug=True)
