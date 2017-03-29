import json
# filePath=r'../data/testres/total_res_dic.txt'
# filePath=r'../data/testres/nonterminal_res_dic.txt'
filePath=r'../data/testres/terminal_res_dic.txt'


stop_words={'4999','0','1'}

with open('../data/testres/id_to_word.txt','r') as f:
    line=f.readline().strip()
    id_to_word=json.loads(line)

with open(filePath,'r') as f:
    total_dic=json.loads(f.readline().strip())
    hit_dic=json.loads(f.readline().strip())
    acc_dic=json.loads(f.readline().strip())

    print(total_dic)
    print(hit_dic)
    print(acc_dic)

    acc_dic=sorted(acc_dic.items(), key=lambda x:x[1])
    total_tokens=0
    for (k,v) in total_dic.items():
        if(k in stop_words):
            continue
        total_tokens+=v
    total_sum=0
    hit_sum=0
    for (k,v) in acc_dic:
        if(k in stop_words):
            continue
        # if(total_dic[k]<50):
        #     continue
        if v >0:
            print(str(id_to_word[k])+' , '+str(v)+" , "+str(total_dic[k]))
            total_sum+=total_dic[k]
            hit_sum+=hit_dic[k]

        # if v>0:
        #     print(str(k)+' , '+str(v))
    print(total_tokens)
    print(total_sum)
    print(hit_sum)
