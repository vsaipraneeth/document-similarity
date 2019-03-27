from flask import Flask, session
from flask import request
from flask import render_template
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from itertools import product
import numpy
import os
import numpy
import os
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import xlsxwriter

UPLOAD_FOLDER = "C:\Python27\MajorProject\data"
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def main():
    return render_template("index.html")

@app.route('/Browse1',methods = ['GET', 'POST'])
def browse1():
    if request.method == 'POST':
        file1  = request.files['file1']
        session['fileName1']=file1.filename
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'],file1.filename))
        print(file1.filename+" uploaded sucessfully")
    return render_template("browse1.html")    

@app.route('/Browse-files',methods = ['GET', 'POST'])
def browseFiles():
    if request.method == 'POST':
        file2  = request.files['file2']
        session['fileName2']=file2.filename
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'],file2.filename))
        print(file2.filename+" uploaded sucessfully")
        with open("data/"+session['fileName1'],"r") as f:
            c1=f.read().replace('\n', '')
        with open("data/"+session['fileName2'],"r") as f1:
            c2=f1.read().replace('\n', '')           
    return render_template("browse-files.html",file1=session['fileName1'],file2=session['fileName2'],content1=c1,content2=c2)

@app.route('/Result', methods=['GET', 'POST'])
def result():
    with open("data/"+session['fileName1'],"r") as f:
            str1=f.read().replace('\n', '')
    with open("data/"+session['fileName2'],"r") as f1:
            str2=f1.read().replace('\n', '') 
    select = request.form.get('alg')
    keywords=request.form.get('keywords')
    keyword=keywords.split(',')
    #print(keyword)
    #print(str(select))
        
    #str1 = "A cemetery is a place where dead people's bodies or their ashes are buried.Ballmer has been vocal in the past warning that Linux is a threat to Microsoft.I was given a card by her in the garden.Many consider Maradona as the best player in soccer history.He loves to play football.Abhishek is a good boy.Cat is drinking water."
    #str2 = "A graveyard is an area of land ,sometimes near a church, where dead people are buried.In the memo, Ballmer reiterated the open-source threat to Microsoft.In the garden, she gave me a card.Maradona is one of the best soccer player.Football is his favourite sport.Football is his favourite sport.Lions eat flesh."

    stop_words = set(stopwords.words("english"))
    stop_words.remove("not")
    stop_words.remove("than")
    
    filtered_sentence1 = []
    filtered_sentence2 = []
    lemm_sentence1 = []
    lemm_sentence2 = []
    sims = []
    temp1 = []
    temp2 = []
    simi = []
    final = []
    same_sent1 = []
    same_sent2 = []
    l=[]
    w1={}
    w2={}
    cs=0
    js=0
    ed=0
    similarity_index=0

    lemmatizer  =  WordNetLemmatizer()

    #print(stop_words)
    for words1 in word_tokenize(str1):
        if words1 not in stop_words:
            if words1.isalnum():
                filtered_sentence1.append(words1)

    for i in filtered_sentence1:
        lemm_sentence1.append(lemmatizer.lemmatize(i))
    

    for words2 in word_tokenize(str2):
        if words2 not in stop_words:
            if words2.isalnum():
                filtered_sentence2.append(words2)


    for i in filtered_sentence2:
        lemm_sentence2.append(lemmatizer.lemmatize(i))

    c=0
    for w in keyword:
        if w in lemm_sentence1:
            w1[c]=1
        else:
            w1[c]=0
        c+=1
    c=0
    for w in keyword:
        if w in lemm_sentence2:
            w2[c]=1
        else:
            w2[c]=0
        c+=1
                    
    if select=="Cosine Similarity":
        #Cosine Similarity
        m=0
        m1=0
        m2=0
        s=0
        cs=0
        for i in w1:
            for j in w2:
                if i==j:
                    s+=w1[i]*w2[j]
        #print("s:",s)
        for i in w1:
            m1+=w1[i]
        for i in w2:
            m2+=w2[i]
        #print("m1:",m1)
        #print("m2:",m2)
        m=math.sqrt(m1)*math.sqrt(m2)
        #print("m:",m)
        cs=s/m
        print("Cosine simialrity is:",cs)
        return render_template("result.html",cs=cs,js=js,ed=ed,sem=similarity_index)
        #return "<h1>Cosine Similarity is:"+str(cs)+"</h1>"

    elif select=="Jaccard Similarity":
        #Jaccard Similarity

        s1=0
        s2=0
        for i in w1:
            for j in w2:
                if i==j:
                    s1+=min(w1[i],w2[j])
        for i in w1:
            for j in w2:
                if i==j:
                    s2+=max(w1[i],w2[j])
        #print("s1:",s1)
        #print("s2:",s2)
        s2=float(s2)
        if s2!=0:
            js=s1/s2
        else:
            js=0
        print("Jaccard Similarity:",js)
        return render_template("result.html",cs=cs,js=js,ed=ed,sem=similarity_index)
        #return "<h1>Jaccard Similarity:"+str(js)+"</h1>"
    
    elif select=="Euclidean Distance":
        #Euclidean Distance
        s=0
        for i in w1:
            for j in w2:
                if i==j:
                    s+=(w1[i]-w2[j])**2
        ed=math.sqrt(s)
        ed=1/(1+ed**0.25)
        ed=round(ed , 2)
        print("Euclidean distance:",ed)
        return render_template("result.html",cs=cs,js=js,ed=ed,sem=similarity_index)
        #return "<h1>Ecilidean Distance:"+str(ed)+"</h1>"

    elif select=="Semantic Similarity":
        #Semantic Similarity
        print("str1:",str1)
        print("str2:",str2)
        print("lem1:",lemm_sentence1)
        print("lem2:",lemm_sentence2)

        c1=0
        if "not" in lemm_sentence1:
            ind=lemm_sentence1.index("not")
            #print("index:",ind)
            lemm_sentence1.remove("not")
            #print(lemm_sentence1)
            dict1={}
            c=0
            for item in lemm_sentence1:
                dict1[c]=str(item)
                c+=1
            #print(dict1)
            for syn in wordnet.synsets(dict1[ind]):
                for l in syn.lemmas():
                    if l.antonyms():
                        if c1==0:
                            dict1[ind]=l.antonyms()[0].name()
                        c1=1
                        #print(l.antonyms()[0].name())
            #print("Before:",lemm_sentence1)
            lemm_sentence1=[]
            for item in dict1.values():
                lemm_sentence1.append(item)
            print("After lemm1:",lemm_sentence1)
        c1=0
        if "not" in lemm_sentence2:
            #print("lemm2:")
            ind=lemm_sentence2.index("not")
            #print("index:",ind)
            lemm_sentence2.remove("not")
            #print(lemm_sentence2)
            dict2={}
            c=0
            for item in lemm_sentence2:
                dict2[c]=str(item)
                c+=1
            #print(dict2)
            for syn in wordnet.synsets(dict2[ind]):
                for l in syn.lemmas():
                    if l.antonyms():
                        if c1==0:
                            dict2[ind]=l.antonyms()[0].name()
                        c1=1
                        #print(l.antonyms()[0].name())
            #print("before:",lemm_sentence2)
            lemm_sentence2=[]
            for item in dict2.values():
                lemm_sentence2.append(item)
            print("After lemm2:",lemm_sentence2)

        for word1 in lemm_sentence1:
            simi =[]
            for word2 in lemm_sentence2:
                sims = []
                syns1 = wordnet.synsets(word1)
                syns2 = wordnet.synsets(word2)
                for sense1, sense2 in product(syns1, syns2):
                    d = wordnet.wup_similarity(sense1, sense2)
                    if d != None:
                        sims.append(d)
    
                if sims != []:        
                   max_sim = max(sims)
                   simi.append(max_sim)
             
            if simi != []:
                max_final = max(simi)
                final.append(max_final)

        similarity_index = numpy.mean(final)
        similarity_index = round(similarity_index , 2)

        count=0
        antonyms=[]
        for word1 in lemm_sentence1:
            for syn in wordnet.synsets(word1):
                for l in syn.lemmas():
                    if l.antonyms():
                        if l.antonyms()[0].name() in lemm_sentence2:
                            if l.antonyms()[0].name() not in antonyms:
                                count+=1
                            #print(count,l.antonyms()[0].name())
                            antonyms.append(l.antonyms()[0].name())                    
                    
        #print("an:",antonyms)
        #print("count:",count)
        if len(antonyms)!=0:
            if count%2!=0:
                final=[0]
                if "than" in lemm_sentence1:
                    if "than" in lemm_sentence2:
                        id1=lemm_sentence1.index("than")
                        id2=lemm_sentence2.index("than")
                        lemm_sentence1.remove("than")
                        lemm_sentence2.remove("than")
                        dict1={}
                        dict2={}
                        c=0
                        for item in lemm_sentence1:
                            dict1[c]=str(item)
                            c+=1
                        c=0
                        for item in lemm_sentence2:
                            dict2[c]=str(item)
                            c+=1
                        #print("id1:",id1,"id2:",id2)
                        #print("1:",dict1,"dict2:",dict2)
                        #print("1:",lemm_sentence1)
                        #print("2:",lemm_sentence2)
                        id1-=1
                        yes=0
                        for syn in wordnet.synsets(dict1[id1]):
                            for l in syn.lemmas():
                                if l.name() in dict2[id2]:
                                    if yes==1:
                                        break
                                    yes=1                         
                        id1+=1
                        id2-=1
                        if yes==1:
                            for syn in wordnet.synsets(dict1[id1]):
                                for l in syn.lemmas():
                                    if l.name() in dict2[id2]:
                                        if yes==2:
                                            break
                                        yes=2
                        if yes==2:
                            final=[1]
            similarity_index = numpy.mean(final)
            similarity_index = round(similarity_index , 2)

        print("Sentence 1: ",str1)
        print("Sentence 2: ",str2)
        print("Similarity index value : ", similarity_index)

        if similarity_index>0.8:
            return "<h1>Similar<br/>Semantic Similarity:"+str(similarity_index)+"</h1>"
        elif similarity_index>=0.6:
            return "<h1>Somewhat Similar<br/>Semantic Similarity:"+str(similarity_index)+"</h1>"
        else:
            return "<h1>Not Similar<br/>Semantic Similarity:"+str(similarity_index)+"</h1>"
    else:
        
        #Cosine Similarity
        m=0
        m1=0
        m2=0
        s=0
        for i in w1:
            for j in w2:
                if i==j:
                    s+=w1[i]*w2[j]
        #print("s:",s)
        for i in w1:
            m1+=w1[i]
        for i in w2:
            m2+=w2[i]
        #print("m1:",m1)
        #print("m2:",m2)
        m=math.sqrt(m1)*math.sqrt(m2)
        #print("m:",m)
        cs=s/m
        print("Cosine simialrity is:",cs)  


        #Jaccard Similarity
        s1=0
        s2=0
        for i in w1:
            for j in w2:
                if i==j:
                    s1+=min(w1[i],w2[j])
        for i in w1:
            for j in w2:
                if i==j:
                    s2+=max(w1[i],w2[j])
        #print("s1:",s1)
        #print("s2:",s2)
        js=s1/float(s2)
        print("Jaccard Similarity:",js)

        #Euclidean Distance
        s=0
        for i in w1:
            for j in w2:
                if i==j:
                    s+=(w1[i]-w2[j])**2
        ed=math.sqrt(s)
        ed=1/(1+ed**0.25)
        ed=round(ed , 2)
        print("Euclidean distance:",ed)

        #Semantic Similarity

        c1=0
        if "not" in lemm_sentence1:
            ind=lemm_sentence1.index("not")
            #print("index:",ind)
            lemm_sentence1.remove("not")
            #print(lemm_sentence1)
            dict1={}
            c=0
            for item in lemm_sentence1:
                dict1[c]=str(item)
                c+=1
            #print(dict1)
            for syn in wordnet.synsets(dict1[ind]):
                for l in syn.lemmas():
                    if l.antonyms():
                        if c1==0:
                            dict1[ind]=l.antonyms()[0].name()
                        c1=1
                        #print(l.antonyms()[0].name())
            #print("Before:",lemm_sentence1)
            lemm_sentence1=[]
            for item in dict1.values():
                lemm_sentence1.append(item)
            #print("After lemm1:",lemm_sentence1)
        c1=0
        if "not" in lemm_sentence2:
            ind=lemm_sentence2.index("not")
            #print("index:",ind)
            lemm_sentence2.remove("not")
            #print(lemm_sentence2)
            dict2={}
            c=0
            for item in lemm_sentence2:
                dict2[c]=str(item)
                c+=1
            #print(dict2)
            for syn in wordnet.synsets(dict2[ind]):
                for l in syn.lemmas():
                    if l.antonyms():
                        if c1==0:
                            dict2[ind]=l.antonyms()[0].name()
                        c1=1
                        #print(l.antonyms()[0].name())
            #print("before:",lemm_sentence2)
            lemm_sentence2=[]
            for item in dict2.values():
                lemm_sentence2.append(item)
            #print("After lemm2:",lemm_sentence2)

        for word1 in lemm_sentence1:
            simi =[]
            for word2 in lemm_sentence2:
                sims = []
                syns1 = wordnet.synsets(word1)
                syns2 = wordnet.synsets(word2)
                for sense1, sense2 in product(syns1, syns2):
                    d = wordnet.wup_similarity(sense1, sense2)
                    if d != None:
                        sims.append(d)
    
                if sims != []:        
                   max_sim = max(sims)
                   simi.append(max_sim)
             
            if simi != []:
                max_final = max(simi)
                final.append(max_final)

        similarity_index = numpy.mean(final)
        similarity_index = round(similarity_index , 2)

        count=0
        antonyms=[]
        for word1 in lemm_sentence1:
            for syn in wordnet.synsets(word1):
                for l in syn.lemmas():
                    if l.antonyms():
                        if l.antonyms()[0].name() in lemm_sentence2:
                            if l.antonyms()[0].name() not in antonyms:
                                count+=1
                            #print(count,l.antonyms()[0].name())
                            antonyms.append(l.antonyms()[0].name())                    
                    
        #print("an:",antonyms)
        #print("count:",count)
        if len(antonyms)!=0:
            if count%2!=0:
                final=[0]
                if "than" in lemm_sentence1:
                    if "than" in lemm_sentence2:
                        id1=lemm_sentence1.index("than")
                        id2=lemm_sentence2.index("than")
                        lemm_sentence1.remove("than")
                        lemm_sentence2.remove("than")
                        dict1={}
                        dict2={}
                        c=0
                        for item in lemm_sentence1:
                            dict1[c]=str(item)
                            c+=1
                        c=0
                        for item in lemm_sentence2:
                            dict2[c]=str(item)
                            c+=1
                        #print("id1:",id1,"id2:",id2)
                        #print("1:",dict1,"dict2:",dict2)
                        #print("1:",lemm_sentence1)
                        #print("2:",lemm_sentence2)
                        id1-=1
                        yes=0
                        for syn in wordnet.synsets(dict1[id1]):
                            for l in syn.lemmas():
                                if l.name() in dict2[id2]:
                                    if yes==1:
                                        break
                                    yes=1                         
                        id1+=1
                        id2-=1
                        if yes==1:
                            for syn in wordnet.synsets(dict1[id1]):
                                for l in syn.lemmas():
                                    if l.name() in dict2[id2]:
                                        if yes==2:
                                            break
                                        yes=2
                        if yes==2:
                            final=[1]
            similarity_index = numpy.mean(final)
            similarity_index = round(similarity_index , 2)

        print("Sentence 1: ",str1)
        print("Sentence 2: ",str2)
        print("Similarity index value : ", similarity_index)

        f = open('data/values.csv','w')
        f.write("Cosine,Jaccard,Euclidean,Semantic")
        f.write("\n")
        res=str(cs)+","+str(js)+","+str(ed)+","+str(similarity_index)
        f.write(res)
        f.close()
        
        left = [1, 2, 3, 4]
        height = []
        df = pd.read_csv('data/values.csv')        
        height.append(df['Cosine'][0])
        height.append(df['Jaccard'][0])
        height.append(df['Euclidean'][0])
        height.append(df['Semantic'][0])

        random_data = height
        data_start_loc = [0, 0]
        data_end_loc = [data_start_loc[0] + len(random_data), 0]

        workbook = xlsxwriter.Workbook('data/graph.xlsx')
        chart = workbook.add_chart({'type': 'column'})
        chart.set_y_axis({'name': 'Similarity Measures'})
        chart.set_x_axis({'name': 'Similarity Algorithms'})
        chart.set_title({'name': 'Document Semantic Similarity'})

        worksheet = workbook.add_worksheet()
        worksheet.write_column(*data_start_loc, data=random_data)

        chart.add_series({
            'values': [worksheet.name] + data_start_loc + data_end_loc,
            'name':'similarityMeasure',
        })
        worksheet.insert_chart('E5', chart)
        workbook.close()
        print("graph generated successfully")        

        tick_label = ['Cosine', 'Jaccard', 'Euclidean', 'Semantic']
        plt.bar(left, height, tick_label = tick_label,
        width = 0.8, color = ['red', 'green','blue','orange'])
        plt.xlabel('Document Similarity Algorithms')
        plt.ylabel('Document Similarity Values')
        plt.title('Similarity Values Graph')
        plt.show()
        return render_template("result.html",cs=cs,js=js,ed=ed,sem=similarity_index)
        #return "<h1>Cosine Similarity:"+str(cs)+"<br/>Jaccard Similarity:"+str(js)+"<br/>Euclidean Distance:"+str(ed)+"<br/>Semantic Similarity:"+str(similarity_index)+"</h1>"
        #return "<table><tr><th>Cosine Similarity:</th><td>"+str(cs)+"</td></tr><tr><th>Jaccard Similarity:</th><td>"+str(js)+"</td></tr><tr><th>Euclidean Distance:</th><td>"+str(ed)+"</td></tr><tr><th>Semantic Similarity:</th><td>"+str(similarity_index)+"</td></tr></table>"

if __name__ == '__main__':
     app.secret_key = 'A0Zr98j/dsyery3737$@!'
     app.run()
     
