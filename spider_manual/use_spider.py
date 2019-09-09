from bs4 import BeautifulSoup
import requests
import re

# ??requests????????BeautifulSoup??html??
s_url = "https://www.baidu.com"
o_header = {
        'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8', 
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive',
        'Referer': 'https://www.baidu.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
} 
n_timeout = 36 # ??????


'''
??????, url????????????
'''
s_local_url = "img.jpg"
resp_stream = requests.get(s_local_url, stream=True, headers=o_header, timeout=n_timeout) #??????   

with open(s_local_url, 'wb', buffering = 1024) as fp:
    fp.write(resp_stream.content)
    fp.flush()          

'''
??html, url??????
'''
# ??get??????
resp = requests.get(s_url, headers=o_header, timeout= n_timeout)


# ???????
resp.encoding = resp.apparent_encoding


# ???html??soup
soup__html = BeautifulSoup(resp, "lxml")   


# ?????id???abc?h
soup__h = soup__html.find("a", id="h")
print(soup__h.text)


# ?????class???abc?<img>
soup__img_s = soup__html.find("img", class_="abc")
for soup__img in soup__img_s:
    print(soup__img["src"], soup__img.text)


# ?????abc???opq?a
soup__a = soup__html.find("a", attrs= {"abc" :"opq"})
print(soup__a.text)


# ?????abc???opq 1, opq 2?a (???)
soup__a = soup__html.find("a", attrs= {"abc" :re.compile(r"opq(\s\w+)?")})
print(soup__a.text)
