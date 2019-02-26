from urllib.parse import urlparse, urljoin
import re
url = 'http://106.186.122.148:8001/up/kbkj/posts?lng=-121.752716&len=50&dist=8826&userid=bc911050-20ae-11e5-89d4-6fa70176b201&lat=38.534737'
parsed_uri = urlparse(url)
print(parsed_uri)
host = parsed_uri.netloc
path = parsed_uri.path
query = parsed_uri.query
float_pattern = re.compile('^.*[0-9]\\.[0-9]*')
if re.match(float_pattern, '123.9') is not None:
    print(re.match(float_pattern, query).group(0))
else:
    print('empty')
query = ''.join([i for i in str(parsed_uri.query) if not i.isdigit()])
print(host + path + '?' + query)
