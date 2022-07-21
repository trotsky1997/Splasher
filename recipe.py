import os
import sys
import threading
import time
import warnings

import numpy as np
import pandas as pd
import pythoncom
import torch
import torch.nn as nn
import torch.nn.functional as F
from console.utils import wait_key
from playsound import playsound

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

rewards = []
q_values = []
discount = 0.75
discount_length = 5
stop_words = []


# def get_stop_words(db, old_stats):
#     # nums = int(old_stats[4]*old_stats[3]*len(db['word']))
#     nums = int((old_stats[2] - 0.03)*len(db['word']))
#     new_db = db.copy()
#     # new_db['right-wrong'] = new_db['right_times'] - new_db['wrong_times']
#     new_db = new_db.sort_values(by='right_times', ascending=False)
#     return new_db['word'].values[:nums].tolist()


def tts(text):
    def read():
        pythoncom.CoInitialize()
        nonlocal text
        try:
            if not os.path.isfile(f'mp3/{text}.mp3'):
                raise Exception(f'{text} not found')
            full_path = os.path.dirname(__file__).replace('\\', '/')
            playsound(f'''{full_path}/mp3/{text}.mp3''', False)
        except:
            x1 = threading.Thread(target=os.system, args=(
                f'''python -c "import pyttsx3;pyttsx3.speak('{text}')"''',))
            x2 = threading.Thread(target=os.system, args=(f'''aspeak -t "{text}" -o "mp3/{text}.mp3" --mp3''',))
            x1.start()
            x2.start()
            x1.join()
            x2.join()
        pythoncom.CoUninitialize()

    x = threading.Thread(target=read)
    x.start()


def load_db(path='./2400_words.csv'):
    db = pd.read_csv(path, sep=",", header=None)
    if len(db.loc[0]) == 2:
        db.columns = ["word", "definition"]
        db['right_times'] = np.zeros(len(db['word']))
        db['wrong_times'] = np.zeros(len(db['word']))
    else:
        db.columns = ["word", "definition", "right_times", "wrong_times"]
    return db


def save_db(db, path='./2400_words.csv'):
    db.to_csv(path, index=False, header=False)


def get_random_word(db):
    return db.sample(1).index


class policy_network(torch.nn.Module):
    def __init__(self, word_nums, attr_nums,):
        super(policy_network, self).__init__()
        self.fc1 = torch.nn.Linear(attr_nums*word_nums, word_nums//2)
        self.fc2 = torch.nn.Linear(word_nums//2, word_nums//4)
        self.miu_layer = torch.nn.Linear(word_nums//4, word_nums)
        self.logsigma_layer = torch.nn.Linear(word_nums//4, word_nums)
        self.gamma = nn.Parameter(torch.randn(1))
        self.beta = nn.Parameter(torch.randn(1))
        self.epsilon = nn.Parameter(torch.randn(1))
        self.scale = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x, mode='train'):
        x = self.norm(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        miu = self.miu_layer(x)
        if mode == 'train':
            logsigma = self.logsigma_layer(x)
            logit = self.scale*torch.randn(miu.shape) * torch.exp(logsigma) + miu
        else:
            logit = miu
        score = torch.softmax(logit, -1)
        return score

    def sample(self, x):
        return torch.argmax(self.forward(x, mode='eval'))

    def norm(self, x):
        x = x / x.sum()
        return x#(x - x.mean())/(x.var() + self.epsilon).sqrt() * self.gamma + self.beta


def warm_up(db, times):
    """
    The warm_up function is used to warm up the database. 
    It is called by the main function and takes two arguments: db, times. 
    db is a connection to a database, times is an integer that specifies how many times you want to execute the query.

    :param db: Store the data that is used in the function
    :param times: Determine how many times the warm_up function will run
    :return: A list of the indices of the random words
    :doc-author: Trelent
    """
    for i in range(times):
        index = get_random_word(db)
        study_step(db, index)


def env_step(db, model, optimizer):
    global rewards,q_values
    score = model(flatten_db(db))
    index = torch.argmax(score)
    # if db['word'].values[index] in stop_words:
    #     return
    q_value = score[index]
    o_old = get_objective_value(db)
    while True:
        ans = wait_key()
        if ans == 'q':
            quit(db, model, optimizer)
        else:
            break
    tts(db['word'].values[index])
    study_step(db, index.item())
    o_new = get_objective_value(db)
    reward = o_new - o_old
    rewards.append(reward)
    rewards = rewards[-discount_length:]
    q_values.append(q_value)
    q_values = q_values[-discount_length:]
    # print(loss.item())
    optimizer_step(optimizer)
    # time.sleep(1.5)
    # tts_lock.release()


losses = []
batch_size = 4


def optimizer_step(optimizer):
    global losses, batch_size,rewards,q_values
    if len(losses) >= batch_size:
        optimizer.zero_grad()
        torch.stack(losses[1:]).sum().backward()
        optimizer.step()
        losses = []
        rewards = []
        q_values = []
    else:
        if len(rewards) >= 2:
            reward = sum(ix * discount ** i for i ,ix in enumerate(rewards))
            loss = get_loss(reward,q_values[0],q_values[1])
            losses.append(loss)

def study_step(db, index):
    print('\n'+db['word'].values[index].center(24, " "))
    # tts(db['word'].values[index])
    while True:
        ans = wait_key()
        if ans == '\r':
            db.loc[index, 'right_times'] += 1
            break
            # reward = -10
        elif ans == 'q':
            quit(db, model, optimizer)
            break
        elif ans == ' ':
            db.loc[index, 'wrong_times'] += 1
            break
        # reward = 20
    print(db['definition'].values[index].center(24, " "))
    tts(db['word'].values[index])
    # return reward


def get_objective_value(db):
    a = (db['right_times'] != 0).sum()/len(db['word'])
    b = ((db['right_times']+db['wrong_times']) != 0).sum()/len(db['word'])
    c = (db['right_times'] > db['wrong_times']).sum() / \
        ((db['right_times']+db['wrong_times']) != 0).sum()
    score = calc_score(db)
    return score*8 + a*4 + b + c*6


def calc_score(db):
    # scale = len(db['word'])/((db['right_times']+db['wrong_times']) != 0).sum()
    return ((db['right_times'] - db['wrong_times']) / (db['right_times'] + db['wrong_times'] + 1) * db['wrong_times']).sum() / db['wrong_times'].sum()


def get_loss(reward, old_q, new_q):
    return (reward+new_q-old_q) ** 2


def flatten_db(db):
    return torch.tensor(db[['wrong_times', 'right_times']].values).flatten().float()


def save_model(model):
    torch.save(model.state_dict(), './model.pkl')


def load_model(model):
    try:
        model.load_state_dict(torch.load('./model.pkl')).float()
        return model
    except Exception:
        return model.float()


def quit(db, model, optimizer):
    save_db(db)
    save_model(model)
    save_optimizer(optimizer)
    print_stats(db)
    sys.exit(0)


old_stats = None


def print_stats(db):
    global old_stats
    a, b, c, d, e, f = db['wrong_times'].sum(), db['right_times'].sum(), (db['right_times'] != 0).sum()/len(db['word']), ((db['right_times']+db['wrong_times'])
                                                                                                                          != 0).sum()/len(db['word']), (db['right_times'] > db['wrong_times']).sum() / ((db['right_times']+db['wrong_times']) != 0).sum(), calc_score(db)

    if old_stats is None:
        print(
            f'''YOUR STAT DATA:\nwrong times:\t{a}\nright times:\t{b}\ntotal progress:\t{c:.2%}\nexpedition progress:\t{d:.2%}\ndive progress:\t{e:.2%}\ntimes weighted score:\t{f:.2%}''')
        old_stats = [a, b, c, d, e, f]
    else:
        print(f'''YOUR STAT DATA:\nwrong times:\t{a}\t{a-old_stats[0]:+}\nright times:\t{b}\t{b-old_stats[1]:+}\ntotal progress:\t{c:.2%}\t{(c-old_stats[2]):+.2%}\nexpedition progress:\t{d:.2%}\t{(d-old_stats[3]):+.2%}\ndive progress:\t{e:.2%}\t{(e-old_stats[4]):+.2%}\ntimes weighted score:\t{f:.2%}\t{(f-old_stats[5]):+.2%}''')
        if c-old_stats[2] > 0:
            print(
                f'''{(b-old_stats[1]) / (c-old_stats[2]) * (1-old_stats[2]) * (db['wrong_times'] + db['right_times']).sum()/db['right_times'].sum()} more recitations are expected''')


def save_optimizer(optimizer):
    torch.save(optimizer.state_dict(), './optimizer.pkl')


def load_optimizer(optimizer):
    try:
        optimizer.load_state_dict(torch.load(
            './optimizer.pkl', map_location='cpu'))
        return optimizer
    except:
        return optimizer


if __name__ == '__main__':
    db = load_db()
    print_stats(db)
    # stop_words = get_stop_words(db, old_stats)
    model = policy_network(len(db['word']), len(db.columns)-2)
    model = load_model(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6,momentum=0.9)
    # optimizer = load_optimizer(optimizer)
    # warm_up(db,10)

    print('press any key to start...')
    while True:
        env_step(db, model, optimizer)
    quit(db, model)
 