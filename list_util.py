
# import json
# from tqdm import tqdm
# words = []
# with open("./皇冠GRE超级词频1.txt", "r") as f:
#     txt = f.read()
    
#     for i in txt.split('''<span class="font37" style="font-weight:bold;">''')[1:]:
#         words.append(i.split('</span>')[0].split('>')[-1])
#     # for i in txt.split('''<span class="font36" style="font-weight:bold;">''')[1:]:
#     #     words.append(i.split('</span>')[0])   

#     print(len(words))



# import sys
# import requests
# from bs4 import BeautifulSoup

# def get_def(search_word='love'):
#     # print(search_word)

#     r = requests.get(f'https://www.bing.com/dict/search?q={search_word}')

#     bs = BeautifulSoup(r.text, 'html.parser')

#     word_types = []
#     definitions = []
#     ans = ''
#     # derivs = []
#     # anathom = []
#     # symthom = []
#     # for e in bs.find_all(class_='hd_if'):
#     #     derivs.append(e.text)
#     # for e in bs.find_all('span', class_='pos'):
#     #     word_types.append(e.text)

#     # for e in bs.find_all('span', class_='def'):
#     #     definitions.append(e.text)
    
#     # for i,j in zip(word_types, definitions):
#     #     if i != '网络' or len(word_types) <= 1:
#     #         ans += i + j
#     # print(derivs)
#     for e in bs.find_all(class_='hd_prUS b_primtxt'):
#         ans += e.text
#     return ans

# defs = []

# for word in tqdm(words):
#     defs.append(get_def(word))

# with open('prnounce.txt',"w") as f:
#     f.write('\n'.join(defs))
#Checking antonym for the word "increase"
#Checking synonym for the word "travel"
from nltk.corpus import wordnet
#Creating a list 
synonyms = []
for syn in wordnet.synsets("sober"):
    for lm in syn.lemmas():
             synonyms.append(lm.name())#adding into synonyms
print (set(synonyms))