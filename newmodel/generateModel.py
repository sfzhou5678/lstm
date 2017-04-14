# -*- coding: utf-8 -*-
#这里确定哪个模型
from __future__ import print_function
from newmodel.basic_ptb import *
from tools import data_reader

def generate(length):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to PTB data directory")
    word_to_id = data_reader.get_word_to_id(FLAGS.data_path)
    id_to_word=data_reader.reverseDic(word_to_id)
    config = get_config()
    # raw_data = data_reader.raw_data(max_data_row=config.max_data_row,data_path=FLAGS.data_path, word_to_id=word_to_id, max_length=config.num_steps)
    # train_data, test_data,voc_size, end_id, _, START_MARK, END_MARK, PAD_MARK = raw_data
    START_MARK = word_to_id['{']
    END_MARK = word_to_id['}']
    PAD_MARK = word_to_id['PAD']
    end_id=word_to_id['ENDMARKER']
    sys.stdout.write("> ")
    sys.stdout.flush()
    token = sys.stdin.readline().strip('\n').split(' ')
    for i in range(len(token)):
        if token[i] not in word_to_id:
            token[i]=word_to_id['UNK']
        else:
            token[i]=word_to_id[token[i]]
    count=0
    res=[]
    while(count<length):
        with tf.Graph().as_default():
            initializer = tf.random_uniform_initializer(-config.init_scale,
                                                        config.init_scale)

            with tf.name_scope("Train"):
                decode_input = PTBInput(config=config, data=token, name="TrainInput",isDecode=True)
                with tf.variable_scope("Model", reuse=None, initializer=initializer):
                    decode_model = PTBModel(is_training=False, config=config, input_=decode_input,
                                            START_MARK=START_MARK, END_MARK=END_MARK, PAD_MARK=PAD_MARK)

                # ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
                # if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                #     print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
                #     # model.saver.restore(session, ckpt.model_checkpoint_path)
                # else:
                #     print("Created model with fresh parameters.")
            Config = tf.ConfigProto()
            Config.gpu_options.allow_growth = True
            sv = tf.train.Supervisor(logdir=FLAGS.save_path)
            with sv.managed_session(config=Config) as session:
                output=run_epoch(session,decode_model,id_to_word,end_id=end_id,isDecode=True)
                # output = decode_model.step(session)
                # print(output)
                tmp = list(output[-1])
                index=tmp.index(max(tmp))
                token.append(index)
                res.append(id_to_word[index])
                count+=1

    for i in range(length):
        print(res[i],end=' ')


generate(20)