"""临时验证脚本：确认已删除端点 404、前端无 academic 残留"""
import urllib.request
import urllib.error

for path in ['/query', '/upload', '/trace']:
    try:
        req = urllib.request.Request('http://localhost:8000' + path, data=b'{}', method='POST')
        urllib.request.urlopen(req, timeout=5)
        print(path, '-> 200 (不该存在!)')
    except urllib.error.HTTPError as e:
        print(path, '->', e.code, '(已删除 OK)')
    except Exception as e:
        print(path, '->', type(e).__name__, e)

html = urllib.request.urlopen('http://localhost:8000/').read().decode('utf-8')
print('前端: academic残留 =', 'academic' in html)
print('前端: scene-switcher残留 =', 'scene-switcher' in html)
print('前端: 名人对话 =', '名人对话' in html)
print('前端: 学术助手残留 =', '学术助手' in html)
print('前端: currentScene persona =', "currentScene: 'persona'" in html)
print('页面长度 =', len(html))
