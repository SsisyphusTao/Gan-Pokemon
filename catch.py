import cv2 as cv
import json
import random
import bs4
import os
import requests

# 内容提取链接
root = 'https://wiki.52poke.com'
url = 'https://wiki.52poke.com/wiki/%E5%AE%9D%E5%8F%AF%E6%A2%A6%E5%88%97%E8%A1%A8%EF%BC%88%E6%8C%89%E5%85%A8%E5%9B%BD%E5%9B%BE%E9%89%B4%E7%BC%96%E5%8F%B7%EF%BC%89/%E7%AE%80%E5%8D%95%E7%89%88'

# 构造头部
ua_list = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Mozilla/5.0 (Windows NT 6.1; rv2.0.1) Gecko/20100101 Firefox/4.0.1",
    "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
    "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11"
]
user_agent = random.choice(ua_list)
headers = {'User-Agent': user_agent}

def getPokedex(url):
    pokedex = []
    with requests.get(url=url, headers=headers) as response:
        data = response.content.decode()
        soup = bs4.BeautifulSoup(data,'lxml')
        title_list = soup.select('.mw-redirect')
    for t in title_list[1:]:
            if t.string[0].encode('utf-8').isalpha():
                content_link = t.get('href')
                sub_url = root + content_link
                pokedex.append(sub_url)
    return pokedex

def getMainPage(urls):
    pokelist = {}
    for i in urls:
        print(i, flush=True)
        with requests.get(url=i, headers=headers) as response:
            data = response.content.decode()
            soup = bs4.BeautifulSoup(data,'lxml')
            title_list = soup.select('.image[href]')
            for a in title_list:
                if i.split('/')[-1] in a.find('img').get('alt'):
                    pokelist[a.get('href').split(':')[-1]]=root+a.get('href')
                    print(a.get('href').split(':')[-1], root+a.get('href'), flush=True)
    with open('Pokedex.json', 'w') as f:
        json.dump(pokelist, f)
    return pokelist

def getDetailContent(url):
    with requests.get(url=url, headers=headers) as response:
        data = response.content.decode()
        soup = bs4.BeautifulSoup(data, 'lxml')
        content = soup.select('.fullImageLink>a')
        for i in content:
            print(i.get('href'), flush=True)
            img_url = i.get('href')
            with requests.get(url='https:'+img_url, headers=headers) as r:
                if 'Dream' in img_url:
                    open('/Users/videopls/Desktop/Gan-Pokemon/collections_dream/'+img_url.split('/')[-1], 'wb').write(r.content)
                else:
                    open('/Users/videopls/Desktop/Gan-Pokemon/collections/'+img_url.split('/')[-1], 'wb').write(r.content)
    return

def mainFunc(t_list):
    for t in t_list:
        content = getDetailContent(t_list[t])

if __name__ == '__main__':
    with open('Pokedex.json', 'r') as f:
        urls = json.load(f)
    mainFunc(getMainPage(urls))