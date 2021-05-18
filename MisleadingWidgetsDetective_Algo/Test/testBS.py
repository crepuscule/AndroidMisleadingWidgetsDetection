def getInfofromApkfab(website):
    # https://apkfab.com/free-apk-download?q=
    # logo: "article-area" > class "package-icon"
    # appname: "article-area" > class "package-name"
    # contact: "package-links" > Issue Tracker
    from bs4 import BeautifulSoup
    from fake_useragent import UserAgent
    import requests
    ua=UserAgent()
    headers={"User-Agent":ua.random}
    proxies = {'http': '127.0.0.1:8118',
     'https': '127.0.0.1:8118'
    }

    html = requests.get(website,headers=headers,proxies=proxies)
    if html.status_code != 200:
        print('status_code:',html.status_code,' error!')
        return None,None,None
    soup = BeautifulSoup(html.content,'lxml')

    main = soup.find('div', {'class', 'packageInfo'})
    logo = main.a.img.get('src')
    # = main.find('a[class="title"]')
    appnames = soup.select('.packageInfo > div > a')
    for item in appnames:
        print('=>',item)
        if item.get('class') == 'title':
            print('#',item.get('string'))
            appname = item.get('string')
            download = item.get('href')
            break
    return logo,appname,'',download

print(getInfofromApkfab('https://apkfab.com/free-apk-download?q=com.hostwr.BestJokesFunny'))
