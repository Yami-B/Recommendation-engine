import csv
import sqlite3
import matplotlib.pyplot as plt
import numpy as np

#Loading dataset train_submissions.csv which includes users ID, the problems ID they worked on
#and the number of attempts made
with open('C:\\Users\\Yasta\\Desktop\\project\\train\\train_submissions.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    user_id=[]
    prob=[]
    attempts_range=[]
    for row in csv_reader:
        user_id.append(int(row[0][5:]))
        prob.append(int(row[1][5:]))
        attempts_range.append(int(row[2]))



#Creating a database named 'users.db' and a table named 'users' of users comprising id, problem solved and 
#corresponding number of attempts
db = sqlite3.connect('users.db')
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS users(
     id INTEGER,
     problems INTEGER,
     attempts_range INTEGER
)
""")
db.commit()
for i in range (0,len(user_id)):
    cursor.execute("""INSERT INTO users(id,problems,attempts_range) VALUES(?,?,?)""", 
                   (user_id[i],prob[i],attempts_range[i]))
db.commit()
db.close()

with open('C:\\Users\\Yasta\\Desktop\\project\\train\\problem_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    problems=[]
    level=[]
    points=[]
    tags=[]
    for row in csv_reader:
        problems.append(int(row[0][5:]))
        if (len(row[4])!=0):
           level.append(int(row[4]))
        else:
           level.append(None)
        if (len(row[5])!=0):
            points.append(int(float(row[5])))
        else:
           points.append(None)
        if (len(row[3])!=0):
            tags.append(str(row[3]))
        else:
            tags.append(None)
        


#Creating a table of problems id, level, points, tags
db = sqlite3.connect('users.db')
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS problems_info(
     problems INTEGER,
     level TEXT,
     points INTEGER,
     tags TEXT)
""")
db.commit()
for i in range (0,len(problems)):
    cursor.execute("""INSERT INTO problems_info(problems,level,points,tags) 
    VALUES(?,?,?,?)""",(problems[i],level[i],points[i],tags[i]))
db.commit()
db.close()
        





#Loading of dataset train_submissions.csv
with open('C:\\Users\\Yasta\\Desktop\\project\\train\\user_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    line_count = 0
    user_id1=[]
    submission_count=[]
    problem_solved=[]
    contribution=[]
    country=[]
    follower_count=[]
    last_online_time_seconds=[]
    max_rating=[]
    rating=[]
    rank=[]
    registration_time_seconds=[]
    for row in csv_reader:
        user_id1.append(int(row[0][5:]))
        submission_count.append(int(row[1]))
        problem_solved.append(int(row[2]))
        contribution.append(int(row[3]))
        if len(country)!=0:
           country.append(str(row[4]))
        else:
           country.append(None)
        follower_count.append(int(row[5]))
        last_online_time_seconds.append(int(row[6]))
        max_rating.append(float(row[7]))
        rating.append(float(row[8]))
        rank.append(str(row[9]))
        registration_time_seconds.append(int(row[10]))



#Creating a table of users containing id and other informations about users
db = sqlite3.connect('users.db')
cursor = db.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS stats(
     id INTEGER,
     submission_count INTEGER,
     problem_solved INTEGER,
     contribution INTEGER,
     country TEXT, follower_count INTEGER,last_online_time_seconds INTEGER,
     max_rating REAL, rating REAL, rank TEXT, registration_time_seconds INTEGER
)
""")
db.commit()
for i in range (0,len(user_id1)):
    cursor.execute("""INSERT INTO stats(id,submission_count,problem_solved,contribution,
                                        country,follower_count,last_online_time_seconds
                                        ,max_rating,rating,rank,registration_time_seconds) VALUES(?,?,?,?,?,?,?,?,?,?,?)""", 
                   (user_id1[i],submission_count[i],problem_solved[i],
                    contribution[i],country[i],follower_count[i], last_online_time_seconds[i],
                    max_rating[i], rating[i],rank[i],registration_time_seconds[i]))
db.commit()
db.close()

#Function which takes as argument an SQL command and executes it
def Command(command):
    db = sqlite3.connect('users.db')
    cursor = db.cursor()
    cursor.execute(command)
    db.commit()
    rows = cursor.fetchall()
    db.close()
    return(rows)
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################




plt.plot(user_id,attempts_range,'ro')
plt.show()

liste_users=[]
for i in range (0,len(user_id)):
    if user_id not in liste_users:
        liste_users.append(user_id[i])

#Selecting performances of each user (number of attempts on each problem solved)       
db = sqlite3.connect('users.db')
cursor = db.cursor()
problems_attempts=[]

cursor.execute("""SELECT problems,attempts_range
FROM users WHERE id IN"""+str(tuple(liste_users))+"""""")
db.commit()
rows = cursor.fetchall()
problems_attempts.append(rows)
db.close()


count=0     
for i in range(0,len(country)):
    if len(country[i])!=0:
        count+=1
print(count/len(country)*100)

#68 % of countries were indicated let's forget about countries for the moment
#since it will only increase the complexity of the model that we want to design

count=0     
for i in range(0,len(contribution)):
    if contribution[i]!=0:
        count+=1
print(count/len(contribution)*100)

#only 30% of users have contributed wheather they're experts, beginners,advanced
#or intermediate. 

import statistics
contributions=[[],[],[],[]]

for i in range (0,len(rank)):
    if rank[i]=='expert' and registration_time_seconds[i]!=0:
        contributions[0].append(registration_time_seconds[i])        
    if rank[i]=='advanced' and registration_time_seconds[i]!=0:
        contributions[1].append(registration_time_seconds[i])        
    if rank[i]=='intermediate' and registration_time_seconds[i]!=0:
        contributions[2].append(registration_time_seconds[i])       
    if rank[i]=='beginner' and registration_time_seconds[i]!=0:
        contributions[3].append(registration_time_seconds[i])
        
avg_contribution=[sum(contributions[0])/len(contributions[0]),
                  sum(contributions[1])/len(contributions[1]),
                  sum(contributions[2])/len(contributions[2]),
                  sum(contributions[3])/len(contributions[3])]
sdv_contribution=[statistics.stdev(contributions[0]),
                  statistics.stdev(contributions[1]),
                  statistics.stdev(contributions[2]),
                  statistics.stdev(contributions[3])]

#Tracing average contributions depending on ranks
plt.rcParams["figure.figsize"] = (5,3)   
objects = ['expert','advanced','intermediate','beginner']
y_pos = np.arange(len(objects))
performance = avg_contribution

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel("Average registration time seconds")
plt.title("Ranks")
plt.show()

#Tracing standard deviation of contributions depending on ranks
plt.rcParams["figure.figsize"] = (5,3)   
objects = ['expert','advanced','intermediate','beginner']
y_pos = np.arange(len(objects))
performance = sdv_contribution

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel("Standard Deviation of resgistration time seconds")
plt.title("Ranks")

plt.show()


#Tracing performances of a given user (number of attempts on each problem solved) 

plt.rcParams["figure.figsize"] = (20,3)
def performances_of_user(user):
    index=liste_users.index(user)
    objects = [j[0] for j in problems_attempts[index]]
    y_pos = np.arange(len(objects))
    performance = [j[1] for j in problems_attempts[index]]
    
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Attempts for User '+ str(liste_users[index]))
    plt.title('Problems')
    
    plt.show()

performances_of_user(232)


#Therefore the higher the user is ranked, the higher is the probability for the user to 
#contribute to judges. Howerver in the higher the rank, the amount of contributions is sparser
#We might as well not take into account contributions



#Work on problem_data in order to impute missing values and predicting number of attempts
level1=[]
points1=[]
tags1=[]
for i in range (0, len(level)):
    if tags[i]!=None:
        level1.append(level[i])
        tags1.append(tags[i].split(","))
classes=[]
for j in range (0,len(tags1)):
    for k in range (0,len(tags1[j])):
        if tags1[j][k] not in classes:
            classes.append(tags1[j][k])

level2=['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
tags_contribution=[[0 for j in range(0,len(classes))] for i in range(0,len(level2))]
for j in range(0,len(tags1)):
    for k in range(0,len(tags1[j])):
        if level1[j]!=0:
            index1=classes.index(tags1[j][k])
            index2=level2.index(str(level1[j]))
            tags_contribution[index2][index1]+=1

for i in range (0,len(classes)):
    objects=level2
    performance=[tags_contribution[j][i] for j in range(0,len(tags_contribution))]
    y_pos=np.arange(len(objects))
    plt.bar(y_pos,performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel("Contribution of tag "+classes[i]+" in levels")
    plt.show()

    

#Calculates lift of each tag to each level
lift=[[0 for j in range(0,len(classes))] for i in range(0,len(level2))]
for i in range(0,len(level2)):
    for j in range(0,len(classes)):
        lift[i][j]=(tags_contribution[i][j]/sum(np.array(tags_contribution)[:,j]))/(sum(tags_contribution[i])/len(level1))

#Stores only the tags which lifts are the most significant (>1) to each level
tags_max_lifts=[[] for j in range(0,len(level2))]
for i in range(0,len(level2)):
    for j in range(0,len(lift[0])):
        if lift[i][j]>=1:
            tags_max_lifts[i].append(classes[j])

#No tag is significant for levels B and C so we take the tags that have the highest   
#lifts
tags_max_lifts[1]=['implementation','greedy','brute force']
tags_max_lifts[2]=['implementation','greedy','dp']

#Completes the empty points and tags for each problem which levels are specified
new_points=[]
new_tags=[]
for i in range(0,len(level)):
    if level[i]!=0:
        if points[i]==None:
           new_points.append(level[i]*500)
        else:
           new_points.append(points[i])
        if tags[i]==None:
            new_tags.append(tags_max_lifts[level[i]-1])
        else:
            new_tags.append(tags[i])

#Let's complete missing levels and tags
import statistics
for j in range(0,len(level)):
    if level[j]==None:
        a=Command("""SELECT level FROM problems_info WHERE problems IN (SELECT problems FROM users WHERE id IN (SELECT id from users WHERE problems="""+str(problems[j])+"""))""")
        b=[]
        for i in range(0,len(a)):
           b.append(int(a[j][0])) 
        if len(b)!=0:
            level[j]=round(statistics.median(b))
        else:    
            level[j]=1
    if tags[j]==None:
        tags[j]=tags_max_lifts[level[j]-1]
    if points[j]==0:
        points[j]=level[j]*500

Attempts=[]
def number_attempts2(User1,Prob1):
    similar_problems=[]
    a=[]
    tags_of_Prob1=tags[problems.index(Prob1)]
    for i in range(0,len(problems)):
        for k in range(0,len(tags_of_Prob1)):
            count=0
            if tags_of_Prob1[k] in tags[i]:
                 count+=1
            if count!=0:
                 a.append(count)
                 similar_problems.append(problems[i])
    if len(similar_problems)!=0:
       a=Command("""SELECT AVG(attempts_range) FROM users WHERE problems IN"""+str(tuple(similar_problems))+"""""")
       return(a[0][0])
    else:
       return(1)

    
User=[]
Prob_id=[]
with open('C:\\Users\\Yasta\\Desktop\\project\\test_submissions.csv') as csv_file:
    csv_reader = csv.reader(csv_file)    
    line_count = 0
    for row in csv_reader:
        User.append(int(row[0][5:]))
        Prob_id.append(int(row[1][5:]))
count=0        
for i in range(0,len(User)):
    Attempts.append(number_attempts2(User[i],Prob_id[i]))
    count+=1
    print(count)
for j in range(0,len(Attempts)):
    if Attempts[j]==None:
        Attempts[j]=1
    else:
        Attempts[j]=round(Attempts[j])
with open('C:\\Users\\Yasta\\Desktop\\project\\soluce.csv','w') as csv_file1:
            wr = csv.writer(csv_file1,lineterminator = '\n')
            for attempt in Attempts:
               wr.writerow([attempt])
    

###################################################################################
###################################################################################

#Data visualization on data about users 
new_rank=[]
for i in range(0,len(rank)):
    if rank[i]=="expert":
        new_rank.append(4)
    if rank[i]=="advanced":
        new_rank.append(3)
    if rank[i]=="intermediate":
        new_rank.append(2)
    if rank[i]=="beginner":
        new_rank.append(1)

success_rate=[]
for i in range(0,len(submission_count)):
    success_rate.append(problem_solved[i]/submission_count[i])
plt.plot(new_rank,success_rate,'bo')
        
#Normalize registration time
def normalize(a):
    norm_a=[]
    max_a=max(a)
    min_a=min(a)
    for i in range(0,len(a)):
        norm_a.append((a[i]-min_a)/(max_a-min_a))
    return(norm_a)
norm_registration_time=normalize(registration_time_seconds)
norm_last_online_time=normalize(last_online_time_seconds)
norm_rating=normalize(rating)

plt.plot(new_rank,norm_registration_time,'bo')
#experts tend to be fresh users 

plt.plot(new_rank,norm_last_online_time,'ro')
#experts rarely connect to the site as the last_online_time is generally 
#greater for experts

plt.plot(new_rank,success_rate,'go')
#the higher the rank the higher the possibility to have a high success rate

norm_problem_solved=normalize(problem_solved)
plt.plot(new_rank,norm_problem_solved,'go')
#the higher the number of problems solved the higher the probability to have  
#a high rank

norm_rating=normalize(rating)
plt.plot(new_rank,norm_rating,'go')
#the higher a rating is for a user, the higher his rank is 

norm_max_rating=normalize(max_rating)
plt.plot(new_rank,norm_max_rating,'go')
#idem as rating but less define

norm_contribution=normalize(contribution)        
plt.plot(success_rate,norm_contribution,'go')
#Below a certain success_rate of 50%,there is rarely any contribution 
#The probability to contribute above this threshold increases

country1=[]
for i in range(0,len(country)):
    if '"'+str(country[i])+'"' not in country1 and country[i]!=None:
        country1.append('"'+country[i]+'"')

Ranks=["expert","advanced","intermediate","beginner"]
ranks_countries=[[0 for j in range(0,len(country1))] for i in range(0,len(Ranks))]
for j in range(0,len(Ranks)):
    for k in range(0,len(ranks_countries[j])):
        ranks_countries[j][k]+=Command("""SELECT COUNT(*) 
        FROM stats WHERE country="""+country1[k]+""" AND rank="""+'"'+Ranks[j]+'"'+""" """)[0][0]

for i in range (0,len(country1)):
    objects=Ranks
    performance=[ranks_countries[j][i] for j in range(0,len(ranks_countries))]
    y_pos=np.arange(len(Ranks))
    plt.bar(y_pos,performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel("Ranks by "+country1[i])
    plt.show()

###CLUSTERING THE USERS 
from sklearn.cluster import KMeans
X=[[] for i in range(0,len(user_id1))]
for j in range(0,len(user_id1)):
    X[j].append(submission_count[j])
    X[j].append(problem_solved[j])
    X[j].append(contribution[j])
    X[j].append(follower_count[j])
    X[j].append(last_online_time_seconds[j])
    X[j].append(max_rating[j])
    X[j].append(rating[j])
    X[j].append(registration_time_seconds[j])
    
def distance(Z,Y):
    distance=0
    for i in range(0,len(Z)):
        distance+=(Z[i]-Y[i])**2
    return(distance**(1/2))
    
from sklearn.metrics import silhouette_score
wws=[]
silhouette=[]
nb_clusters=[]
for i in range(2,10):
     
      kmeans = KMeans(n_clusters=i,random_state=0).fit(np.array(X))
      sum_square=[0 for j in range(0,i)]
      for k in range(0,len(user_id1)):
          dist=distance(X[k],kmeans.cluster_centers_[kmeans.labels_[k]])
          sum_square[kmeans.labels_[k]]+=dist
      wws.append(sum(sum_square)/len(sum_square))
      nb_clusters.append(i)
      print(i)
      cluster_labels = kmeans.fit_predict(X)
      silhouette_avg = silhouette_score(X, cluster_labels)
      silhouette.append(silhouette_avg)

plt.plot(nb_clusters,wws,'bo')

plt.plot(nb_clusters,silhouette,'ro')

#6 clusters

kmeans = KMeans(n_clusters=6,random_state=0).fit(np.array(X))

X1=[]
for j in range(0,len(user_id)):
    a=Command("""SELECT submission_count,problem_solved,contribution,follower_count,last_online_time_seconds
                                        ,max_rating,rating,registration_time_seconds FROM stats WHERE id="""+str(user_id[j])+"""""")
    X1.append(kmeans.predict(list(a)))

y=attempts_range

plt.plot(X1,y,'ro')



#Creating matrix of user clusters
matrix=[[]for i in range(0,6)]
for j in range(0,len(user_id1)):
    if kmeans.predict([X[j]])==0:
        matrix[0].append(X[j][0])
    if kmeans.predict([X[j]])==1:
        matrix[1].append(X[j][0])
    if kmeans.predict([X[j]])==2:
        matrix[2].append(X[j][0])
    if kmeans.predict([X[j]])==3:
        matrix[3].append(X[j][0])
    if kmeans.predict([X[j]])==4:
        matrix[4].append(X[j][0])
    if kmeans.predict([X[j]])==5:
        matrix[5].append(X[j][0])



#Predicting attempts of a user thanks to the cluster of users it belongs to
def number_attempts(user,prob_id):
    nb_attempts=[]
    a=Command("""SELECT submission_count,problem_solved,contribution,follower_count,last_online_time_seconds
                                        ,max_rating,rating,registration_time_seconds FROM stats WHERE id="""+str(user)+"""""")
    user_class=kmeans.predict(list(a))
    similar_users=matrix[user_class[0]]
    b=Command("""SELECT attempts_range FROM users WHERE id IN """+str(tuple(similar_users))+"""
                  AND problems="""+str(prob_id)+"""""")
    if b!=[]:
        for k in range(0,len(b)):
            nb_attempts.append(int(b[k][0]))
    if len(nb_attempts)==0:
        return(1)
    else:
        return(round(sum(nb_attempts)/len(nb_attempts)))

attempts=[]
User=[]
Prob_id=[]
count=0
with open('C:\\Users\\Yasta\\Desktop\\project\\test_submissions.csv') as csv_file:
    csv_reader = csv.reader(csv_file)    
    line_count = 0
    for row in csv_reader:
        User.append(int(row[0][5:]))
        Prob_id.append(int(row[1][5:]))
        
for i in range(0,len(User)):
    attempts.append(number_attempts(User[i],Prob_id[i]))
    count+=1
    print(count)
with open('C:\\Users\\Yasta\\Desktop\\project\\hello.csv','w') as csv_file1:
            wr = csv.writer(csv_file1,lineterminator = '\n')
            for attempt in Attempts:
               wr.writerow([attempt])
    
            
            
        
#÷test hypothese mu1=mu2=...=mu6