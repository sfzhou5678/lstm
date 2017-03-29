import json
from collections import Counter


res_path = r'../data/testres/testres.txt'

total_dic = {}
hit_dic = {}
acc_dic={}

topN=10
with open(res_path, 'r') as f:
    while True:
        true_output_line = f.readline().strip()
        if (true_output_line == ''): break
        pred_output_line = f.readline().strip()
        true_output = json.loads(true_output_line)
        pred_output = json.loads(pred_output_line)

        for i in range(len(true_output)):
            target_token = true_output[i]
            if target_token in total_dic:
                total_dic[target_token] += 1
            else:
                total_dic[target_token] = 1

            if(target_token in [item for item in pred_output[i][:topN]]):
                if (target_token in hit_dic):
                    hit_dic[target_token] += 1
                else:
                    hit_dic[target_token] = 1

    for k in total_dic:
        if k in hit_dic:
            acc_dic[k]=hit_dic[k]/total_dic[k]
        else:
            acc_dic[k]=0

    with open(r'../data/testres/total_res_dic.txt','w') as wf:
        wf.write(json.dumps(total_dic)+'\n')
        wf.write(json.dumps(hit_dic)+'\n')
        wf.write(json.dumps(acc_dic)+'\n')


    # print(line2)
