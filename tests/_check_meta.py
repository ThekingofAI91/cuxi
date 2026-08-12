# -*- coding: utf-8 -*-
"""验证 /conversation/{sid}/meta 端点是否已生效（需后端重启后才有）"""
import urllib.request

try:
    r = urllib.request.urlopen('http://localhost:8000/conversation/test123/meta', timeout=10)
    print('META_HTTP:', r.status, r.read().decode('utf-8')[:200])
except Exception as e:
    print('META_HTTP_ERR:', str(e)[:200])
