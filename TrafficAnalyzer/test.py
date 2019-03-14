from TrafficAnalyzer.analyzer import Analyzer, preprocess
from statistics import jaccard

url_a = '/m/getProvinceCode[\\?]latitude=[\\+]&longitude=[\\+]'
a = Analyzer.url_pattern2set(url_a)

url_b = '/mines[\\.]json[\\?]latitude=[\\+]&longitude=[\\+]'
b = Analyzer.url_pattern2set(url_b)

url = 'http://ynuf.alipay.com/m/um.htm?c=%7B%22service%22%3A%22com.taobao.tdp%22%2C%22timestamp%22%3A%221436378080624%22%2C%22app%22%3A%2223177793%22%2C%22os%22%3A%22android%22%2C%22payload%22%3A%22LBqyR3pSW7GyTt0vf66b0pV9fsKunmnG74przzbKAzr6cI0OYI%5C%2FRpCHyjtm0bNydLfVhbWSWvF1vbtUd8PDOC6H9HDvUPBd1rZZPjMtlJSlLFS4onYIGTPVM6rti6lI025gfuUAH%5C%2FbICO1oH21rfqikkKzs2TAL1EZiI%5C%2Fh52sqMY2ctNTSp4r4sb51FuDlSh7pRMb56ZJu%5C%2FDYf41Qk7fBtmaWjO0Da9D83FmVfmS6rCxOyIG9t7%5C%2Fit3PEvTZu45v6NecpFQ5HqFFlZ8c0XfCCnXBOXVCNlS%2BOYnUf9sFtYLavEFgfMM%5C%2FJJg3WfIA9YSzArcw8uzVzL0aSq3EBb89dFCWEkBFW447UYv8resn35Ge6dVSWG1IYuTtRGgcSGqTtcE7XLIWO9DkUudnGhJNnQqA4cL6zJduWqTFGXpop00D5WRBWXjm7j554EMmaIatBSkyrk7BJJ3a3Lz9iFydKEdqdpPX%5C%2Ff%5C%2F3OGjelM5GKs8E96HRy7yicmMqoRNyP2KKn5QgoiJMpYBfAcDwuNCYFkrExM6dZ%5C%2F0IyO9TqYDI3HXWHPWosxUuPOG0n47j83WuCBdjcxvAYrc%5C%2FCi6CbAUlN2USt4tmN6MB8sApcDoNZBnw1kmaRgDxdUbfAmJ%5C%2FqFeitcghU52oJkFcVm77CRvRSzfZdEL7dooKuIJQVihDNr%2BC88fqhy5rsu%5C%2F55yCwQfPoo%5C%2F%2BX7FejaUgh4jF2Xnv7Tboo0L9jyMeZmSp%5C%2FLHPQ2eYbh7QRaUSwksreG8Yt%2BBeMoy7MohT2xVWdv7VTMQ77VpabsrldOW%5C%2FnETZszYHNUNYMMYcWpeFRXN8ds3yd2tVyFeWeOepjGTAt9Y6IxHncVsrGVMbjmvwTG3Tq6MiJKs06uDCPIZegdMXv3br%2BuUB%2Ba86VQB54DyGEPZ3d%5C%2Fs0S6N47%5C%2FuGbEDI6GiAegEZNz96lecD3k6NLrGcsXMip%2BkVHoSOPAgF7%5C%2Fh1x2vJievyVLFU44WUuS%2BHcNAZuhHuxz7s%3D%22%2C%22signature%22%3A%223468bd6e846fc976b6c7601c607767de3d554429%22%2C%22version%22%3A%221.0.1%22%7D'
u = Analyzer.url_pattern2set(url)
print(u)
s = {'htm', 'm', 'um', 'c'}
print(len(u.intersection(s)), len(s))

# print(jaccard(a, b))
# print(a.intersection(b))
#
#
# neg_pcap_dir = '/home/workspace/FlowIntent/data/Location/p2'
# pos_flows, neg_flows = preprocess(neg_pcap_dir)
# Z, pc = Analyzer.signature_dendrogram(pos_flows)
# from scipy.cluster.hierarchy import fcluster
# clusters = fcluster(Z, 0.6671, criterion='distance')
# cls = dict()
# i = 0
# for c in clusters:
#     print(i, c - 1, pc[i])
#     if c not in cls:
#         cls[c] = []
#     cls[c].append(pc[i])
#     i += 1
# print(cls)
# sig_cls = []
# for sigs in cls.values():
#     print(sigs)
#     s = Analyzer.url_pattern2set(sigs[0])
#     for i in range(1, len(sigs)):
#         s = s.intersection(Analyzer.url_pattern2set(sigs[i]))
#     sig_cls.append(s)
#     print(s)
# print(len(sig_cls))
