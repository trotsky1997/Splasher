import os
import random
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
q_next_values = []
discount = 0.75
discount_length = 20
stop_words = []


def get_stop_words(db, old_stats):
    # nums = int(old_stats[4]*old_stats[3]*len(db['word']))
    nums = int((old_stats[3] *old_stats[4] - 0.01)*len(db['word']))
    new_db = db.copy()
    new_db['right/wrong'] = (new_db['right_times']-1) / (new_db['wrong_times'] + new_db['right_times'] + 1)
    new_db = new_db.sort_values(by='right/wrong', ascending=False)
    return new_db.index[:nums].tolist()


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
    db2 = db.copy()
    db2['acc'] = (db2['right_times']) / (db2['right_times'] + db2['wrong_times'] + 1)
    db2.to_excel(path.replace('.csv', '.xlsx'), index=False, header=False)

def get_random_word(db):
    return db.sample(1).index


class policy_network(torch.nn.Module):
    def __init__(self, word_nums, attr_nums,):
        super(policy_network, self).__init__()
        self.fc1 = nn.Linear(3,3)
        self.fc2 = nn.Linear(3,3)
        self.fc3 = nn.Linear(3,1)
        self.miu = nn.Linear(2400,2400)
        self.logsigma = nn.Linear(2400,2400)
        self.masker = nn.Linear(2400,2400)
        self.decoder = nn.Linear(2400,2400)

    def scaled_dot_attention(self,x):
        return torch.softmax(x.mm(x.transpose(-1,-2))/ x.shape[0] ** 0.5,-1).mm(x)

    def forward(self, x1,x2, mode='train'):
        x3 = (x1-x2)/(x1+x2+1.0)
        x3 = self.norm(x3,summ=False).unsqueeze(1)
        x1 = self.norm(x1).unsqueeze(1)
        x2 = self.norm(x2).unsqueeze(1)
        x = torch.cat([x1,x2,x3], dim=-1)
        x = F.relu(self.fc1(x))
        x = x + self.scaled_dot_attention(x)
        x = F.relu(self.fc2(x))
        x_pre = x = F.relu(self.fc3(x)).squeeze()
        # x = torch.softmax(self.fc3(x).squeeze(),-1)
        miu = self.miu(x)
        logsigma = self.logsigma(x)
        mask = torch.sigmoid(self.masker(x))
        noisy = torch.randn(x.shape)
        if self.training:
            x_sample = miu + noisy*torch.exp(logsigma)
        else:
            x_sample = miu
        x_decode = F.relu(self.decoder(x_sample))
        score = torch.softmax(x_decode,-1) * mask
        return score,F.mse_loss(x_pre,x_decode),-F.kl_div(x_sample,noisy)

    def norm(self, x,summ=True):
        if summ:
            x = x / x.sum()
        return (x - x.mean())/x.std()


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

index_old = None
def env_step(db, model, optimizer):
    global rewards,q_values,index_old,q_next_values,stop_words_indexes,additive_losses
    score,rec_loss,kl_loss = model(*flatten_db(db))
    # s_temp = score.detach().data.clone()
    # s_temp[stop_words_indexes] = -1
    index = torch.argmax(score)
    q_value = score[index]
    o_old = get_objective_value(db)
    
    while True:
        ans = wait_key()
        if ans == 'q':
            quit(db, model, optimizer)
        elif ans == 't' and not (index_old is None):
            hint = input(f"type hint for {db['word'].values[index_old]}:")
            if hint != '':
                db.loc[index_old.item(), 'definition'] += "hint:"+hint
                save_db(db)
                break
            else:
                break
        else:
            break
    tts(db['word'].values[index])
    study_step(db, index.item())
    o_new = get_objective_value(db)
    reward = o_new - o_old
    rewards.append(reward)
    q_values.append(q_value)
    with torch.no_grad():
        model.eval()
        scores,_,_ = model(*flatten_db(db))
        q_next_value = scores[index]
        q_next_values.append(q_next_value)
        model.train()
    # print(loss.item())
    additive_losses.append(rec_loss+kl_loss)
    optimizer_step(optimizer)
    index_old = index
    # time.sleep(1.5)
    # tts_lock.release()


losses = []
additive_losses = []
batch_size = 1


def optimizer_step(optimizer):
    global losses, batch_size,rewards,q_values,q_next_values,additive_losses
    if len(losses) >= batch_size:
        optimizer.zero_grad()
        loss = torch.stack(losses).sum()
        loss.backward()
        optimizer.step()
        losses = []
        q_values = []
        rewards = []
        q_next_values = []
        additive_losses = []
    else:
        if len(q_next_values) >= 1:
            reward = sum(ix * discount ** i for i ,ix in enumerate(rewards))
            loss = get_loss(reward,q_values[0],q_next_values[0])+additive_losses[0]
            losses.append(loss)
            q_values = q_values[1:]
            rewards = rewards[1:]
            q_next_values = q_next_values[1:]
            additive_losses = additive_losses[1:]

counter = 0
def study_step(db, index):
    global counter,batchsize
    # calculate the sorted index of the words wrong_times
    rank = db['wrong_times'].rank(pct=True,ascending=True)[index]
    acc = db['right_times'].values[index]/(1 + db['right_times'].values[index] + db['wrong_times'].values[index])
    print('\n'+db['word'].values[index].center(24, " "),f'({counter}/{batch_size}|{acc:.2%}|{rank:.2%})')
    counter = (counter + 1)% batch_size 
    # tts(db['word'].values[index])
    while True:
        ans = wait_key()
        if ans == '\r':
            db.loc[index, 'right_times'] = db.loc[index, 'right_times'] * 0.9 + 1
            db.loc[index, 'wrong_times'] = db.loc[index, 'wrong_times'] *0.9
            break
            # reward = -10
        elif ans == 'q':
            quit(db, model, optimizer)
            break
        elif ans == 't':
            print("hint:",random.choice(db['definition'].values[index].split("hint:")).center(24, " "))
        elif ans == ' ':
            db.loc[index, 'right_times'] = db.loc[index, 'right_times'] * 0.9
            db.loc[index, 'wrong_times'] = db.loc[index, 'wrong_times'] *0.9 + 1
            break
        # reward = 20
    print(db['definition'].values[index].split("hint:")[0].center(24, " "))
    if len(db['definition'].values[index].split("hint:")) > 1:
        print("hint:",random.choice(db['definition'].values[index].split("hint:")[1:]).center(24, " "))
    tts(db['word'].values[index])
    # return reward


def get_objective_value(db):
    a = (db['right_times'] > 0).sum()/len(db['word'])
    b = ((db['right_times']+db['wrong_times']) != 0).sum()/len(db['word'])
    c = (db['right_times'] > db['wrong_times']).sum() / \
        ((db['right_times']+db['wrong_times']) != 0).sum()
    score = calc_score(db)
    return (score*10+a*8+b+c)*10000


def calc_score(db):
    # scale = ((db['right_times']+db['wrong_times']) != 0).sum() / len(db['word'])
    score =  ((db['right_times']) / (db['right_times'] + db['wrong_times'] + 1) * (db['wrong_times']+1)).sum() / (db['wrong_times'].sum() + len(db['word']))
    return score


def get_loss(reward, old_q, new_q):
    return (reward+new_q-old_q) ** 2


def flatten_db(db):
    return torch.tensor(db[['right_times']].values).flatten().float(),torch.tensor(db[['wrong_times']].values).flatten().float()


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
    a, b, c, d, e, f = db['wrong_times'].sum(), db['right_times'].sum(), (db['right_times'] > 1).sum()/len(db['word']), ((db['right_times']+db['wrong_times'])
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

    stop_words_indexes = get_stop_words(db, old_stats)
    model = policy_network(len(db['word']), len(db.columns)-2)
    model = load_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6,weight_decay=1e-4,amsgrad=True)
    # optimizer = load_optimizer(optimizer)
    # warm_up(db,10)

    print('press any key to start...')
    while True:
        env_step(db, model, optimizer)
    quit(db, model) 