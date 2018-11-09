import re
f=open(r'./dict_cn.txt','r')
raw=f.readlines()
f.close()
dictionary=[re.split(r' |\n',w)[0] for w in raw]
max_len=max([len(w) for w in dictionary])

#sent='计算机科学与技术'
sent='在这阳光灿烂的日子'
results=[]
while 1:
    words=sent[:max_len]
    if len(words)==0:
        break
    while 1:
        if words in dictionary:
            results.append(words)
            sent=sent[len(words):]
            break
        else:
            words=words[:-1]
            if len(words)==1:
                results.append(words)
                sent=sent[len(words):]
                break
     
print(results)
