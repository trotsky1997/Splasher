
import sys
import requests
from bs4 import BeautifulSoup

def get_def(search_word='love'):
    # print(search_word)

    r = requests.get(f'https://www.youdao.com/result?word={search_word}&lang=en')

    bs = BeautifulSoup(r.text, 'html.parser')

    poses = []
    transes = []

    for pos in bs.find_all(class_='pos'):
        poses.append(pos.string)

    for trans in bs.find_all(class_='trans'):
        transes.append(trans.string)

    for i,j in zip(poses,transes):
        print(i,j)

    



print(get_def())