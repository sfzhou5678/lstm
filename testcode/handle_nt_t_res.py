import json
filePath=r'../data/testres/total_res_dic.txt'

stop_words={'{','}','NameConstant','PAD'}
stop_terminal_set={'AsName','ModuleName','ClassName','FuncName','AsyncFuncName'}
terminal_set= {'Num','Arg','Str','Bytes','True','False',
               'Name','AttrName','Key'}

def is_terminal(token):
    """
    处理终结符，终结符要么是terminal_set第一行中的特殊符号，或者是如Name('a')这种带括号的形式
    :param token:
    :param isTerminalSet:
    :return:
    """
    type=token[:token.index('(')] if token.endswith(')') else token
    if type in terminal_set:
        return True
    return False


def is_nonterminal(token):
    """
    处理终结符，终结符要么是terminal_set第一行中的特殊符号，或者是如Name('a')这种带括号的形式
    :param token:
    :param isTerminalSet:
    :return:
    """
    if token in terminal_set:
        return False
    if(token.endswith(')') and token.find(r'(')>=0):
        return False
    return True

with open('../data/testres/id_to_word.txt','r') as f:
    line=f.readline().strip()
    id_to_word=json.loads(line)

with open(filePath,'r') as f:
    total_dic=json.loads(f.readline().strip())
    hit_dic=json.loads(f.readline().strip())
    acc_dic=json.loads(f.readline().strip())
    acc_dic=sorted(acc_dic.items(), key=lambda x:x[1])

    nt_total_count=0
    nt_total_dic={}
    nt_hit_dic={}
    nt_acc_dic={}

    t_total_count=0
    t_total_dic={}
    t_hit_dic={}
    t_acc_dic={}

    total_tokens_count=0
    for (k,v) in total_dic.items():
        token=id_to_word[k]
        if token in stop_words:
            continue
        if(k in hit_dic):
            hit=hit_dic[k]
        else:
            hit=0
        if(is_terminal(token)):
            if(k in t_total_dic):
                t_total_dic[k]+=v
            else:
                t_total_dic[k]=v

            if(k in t_hit_dic):
                t_hit_dic[k]+=hit
            else:
                t_hit_dic[k]=hit
            t_total_count+=v
        elif(is_nonterminal(token)):
            if(k in nt_total_dic):
                nt_total_dic[k]+=v
            else:
                nt_total_dic[k]=v

            if(k in nt_hit_dic):
                nt_hit_dic[k]+=hit
            else:
                nt_hit_dic[k]=hit
            nt_total_count+=v
        else:
            print(token)
        total_tokens_count+=v
    print('t:  '+str(t_total_count))
    print('nt: '+str(nt_total_count))
    print('total: '+str(total_tokens_count))

    for k in t_total_dic:
        if k in t_hit_dic:
            t_acc_dic[k]=t_hit_dic[k]/t_total_dic[k]
        else:
            t_acc_dic[k]=0

    with open(r'../data/testres/terminal_res_dic.txt','w') as wf:
        wf.write(json.dumps(t_total_dic)+'\n')
        wf.write(json.dumps(t_hit_dic)+'\n')
        wf.write(json.dumps(t_acc_dic)+'\n')

    for k in nt_total_dic:
        if k in nt_hit_dic:
            nt_acc_dic[k]=nt_hit_dic[k]/nt_total_dic[k]
        else:
            nt_acc_dic[k]=0

    with open(r'../data/testres/nonterminal_res_dic.txt','w') as wf:
        wf.write(json.dumps(nt_total_dic)+'\n')
        wf.write(json.dumps(nt_hit_dic)+'\n')
        wf.write(json.dumps(nt_acc_dic)+'\n')

    # total_sum=0
    # hit_sum=0
    # for (k,v) in acc_dic:
    #     if(k in stop_words):
    #         continue
    #     if(total_dic[k]<50):
    #         continue
    #     if v >0.8:
    #         print(str(id_to_word[k])+' , '+str(v)+" , "+str(total_dic[k]))
    #         total_sum+=total_dic[k]
    #         hit_sum+=hit_dic[k]

            # if v>0:
            #     print(str(k)+' , '+str(v))
    # print(total_sum)
    # print(hit_sum)