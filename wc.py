import json
from os import path
from PIL import Image
import numpy as np
from bs4 import BeautifulSoup
import requests
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
punctuation = '''''!()-[]{};:'"\,<>./?@#$%^&*_~'''
url=input("enter the url:")
respond = requests.get(url)
soup = BeautifulSoup(respond.text,"lxml")
l = soup.find_all('p')
t=str(l)
stop_words=set(stopwords.words('english'))
token=word_tokenize(t)
filtered=[]
for word in token:
    if word not in stop_words and  word not in punctuation:
        filtered.append(word)
filtered = {filtered[i]: filtered[i+1] for i in range(0, len(filtered), 2)}
filtered = {'filtered': filtered}
with open('wc.txt', 'w') as file:
     file.write(json.dumps(filtered))
d=path.dirname(__file__)
text=open(path.join(d,'wc.txt')).read()
virat_mask=np.array(Image.open(path.join(d, 'virat_mask.JPG')))
stopwords=set(STOPWORDS)
stopwords.add("said")
wc = WordCloud(background_color="red",max_words=5000,mask=virat_mask,stopwords=stopwords)
wc.generate(text)
wc.to_file(path.join(d,"vk_mask.png"))
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.figure()
plt.imshow(virat_mask,cmap=plt.cm.gray,interpolation="bilinear")
plt.axis("off")
plt.show()
