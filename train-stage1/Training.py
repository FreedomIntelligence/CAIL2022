import torch
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
import datetime
import pickle

from tqdm import tqdm
import os

learning_rate = 1e-5
running_epoch = 4


def train(model, train_data, save_path):
    optimizer = Adam(model.parameters(), lr=learning_rate, eps=1e-8)
    if os.path.exists(save_path + "saved_model"):
        pass
    else:
        os.mkdir(save_path + "saved_model")
    for epoch in range(running_epoch):
        start_t = datetime.datetime.now()
        print('Epoch[{}/{}]'.format(epoch + 1, running_epoch))
        model.train()
        for item in tqdm(train_data):
            sentence = item[0].cuda()
            sep = item[1].cuda()
            label = item[2].squeeze(0).cuda()
            class_out = model(sentence, sep)
            loss = F.cross_entropy(class_out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), save_path + 'saved_model/epoch{my_epoch}.ckpt'.format(my_epoch=epoch))
        end_t = datetime.datetime.now()
        print('one epoch cost_time:', end_t - start_t)
        # print('Training data:')
        # evaluate(model,train_data)
        # print('Test data:')
        # evaluate(model,test_data)


def evaluate(model, valid_data):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for item in valid_data:
            sentence = item[0].cuda()
            sep = item[1].cuda()
            label = item[2].squeeze(0).cuda()
            class_out = model(sentence, sep)
            loss = F.cross_entropy(class_out, label)
            loss_total += loss
            label = label.cpu().numpy()
            predict = torch.max(class_out.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, label)
            predict_all = np.append(predict_all, predict)
    acc = metrics.accuracy_score(labels_all, predict_all)
    ave_loss = loss_total / len(valid_data)
    report = metrics.classification_report(labels_all, predict_all, digits=4)
    confusion = metrics.confusion_matrix(labels_all, predict_all)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(ave_loss, acc))
    print('Precision, Recall and F1-Score')
    print(report)
    print('Confusion Maxtrix')
    print(confusion)
    print('*' * 20)
    return acc, ave_loss


def test(model, test_data, model_path):
    for i in range(4):
        print('epcoh:', i)
        model.load_state_dict(torch.load(model_path + 'saved_model/epoch{num}.ckpt'.format(num=i)))
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for item in tqdm(test_data):
                sentence = item[0].cuda()
                sep = item[1].cuda()
                label = item[2].squeeze(0).cuda()
                class_out = model(sentence, sep)
                loss = F.cross_entropy(class_out, label)
                loss_total += loss
                label = label.cpu().numpy()
                predict = torch.max(class_out.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, label)
                predict_all = np.append(predict_all, predict)
        acc = metrics.accuracy_score(labels_all, predict_all)
        ave_loss = loss_total / len(test_data)
        report = metrics.classification_report(labels_all, predict_all, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
        print(msg.format(ave_loss, acc))
        print('Precision, Recall and F1-Score')
        print(report)
        print('Confusion Maxtrix')
        print(confusion)
        print('*' * 20)


def get_result(model, evaluate_data,model_path, best_model_index):
    model.load_state_dict(torch.load(model_path+'/saved_model/epoch{num}.ckpt'.format(num=best_model_index)))
    model.eval()
    predict_all = []
    with torch.no_grad():
        for item in evaluate_data:
            sentence = item[0].cuda()
            sep = item[1].cuda()
            class_out = model(sentence, sep)
            predict = torch.max(class_out.data, 1)[1].cpu().numpy()
            predict_all.append(predict)
    print(len(predict_all))
    print(predict_all[0])
    with open(model_path+'/label_out.pickle', 'wb') as f:
        string = pickle.dumps(predict_all)
        f.write(string)
