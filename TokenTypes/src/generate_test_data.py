# encoding:utf-8


stop_words={'{','}','NameConstant'}
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

def handle_line(line, wf, num_steps=None,max_words_length=None,record_time_step=None,isTerminalSet=True):
    line=line.strip()
    cur_line_data=""
    count=0
    tokens=line.split(" ")
    if(num_steps and len(tokens)>num_steps):
        return
    for token in tokens:
        count+=1
        if count==1:
            cur_line_data+=token
        else:
            cur_line_data+=" "+token
        if(max_words_length and count>max_words_length):
            break
        if record_time_step is None or(count%record_time_step==0):
            if token in stop_words:
                continue
            if(is_terminal(token) and isTerminalSet and count>10):
                wf.write(cur_line_data+"\n")
            elif(is_nonterminal(token) and not isTerminalSet and count>10):
                wf.write(cur_line_data+"\n")


original_data_path=r'D:\MyProjectsRepertory\Python_project\s-lstm\data\train.txt'

is_terminal_set=False

num_steps=60
max_words_length=60

save_path=r'train_terminal-%d-%d.txt'%(num_steps,max_words_length) if is_terminal_set else r'train_nonterminal-%d-%d.txt'%(num_steps,max_words_length)
# save_path=r'test_terminal.txt' if is_terminal_set else r'test_nonterminal.txt'

count=0
with open(save_path,'w') as wf:
    with open(original_data_path) as f:
        for line in f:
            handle_line(line, wf, num_steps=num_steps,max_words_length=max_words_length,record_time_step=None,isTerminalSet=is_terminal_set)
            count+=1
            # if count>10:
            #     break



