import argparse

from torch.utils.data import DataLoader
from pathology_dataset import patchDataset_add_age
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from efficientnet_pytorch import EfficientNet
import random
from timeit import default_timer as timer
from sklearn import metrics
import logging

from torch.nn.parameter import Parameter

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else:
        raise NotImplementedError
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b2')

        self.avg_pool =  GeM()


        self.fc = nn.Sequential(nn.Dropout(),
                                    nn.Linear(self.enet._fc.in_features+1, 2))

        self.myfc=nn.Sequential(
            nn.Dropout(0),
            nn.Linear(self.enet._fc.in_features+1, 1),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x,age):

        age=torch.unsqueeze(age,1)
        x = self.enet.extract_features(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x=torch.cat((x,age),1)
        x_reg = self.myfc(x)

        x_cls=self.fc(x)

        return x_reg,x_cls






def get_model(model):
    out_dim = 1

    enet_type = 'efficientnet-b2'

    net = enetv2(enet_type, out_dim=out_dim)
    return net




def run_train(config):


    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG)

    log1 = logging.getLogger()


    #
    train_list = 'spilt/train_1.txt'

    val_list = 'spilt/valid_1.txt'



    train_all_df=pd.read_csv(r'./training_data_classification_labels2020.csv')


    subjects = open(train_list).read().splitlines()
    train_names = [sub.split(" ")[0].split('/')[-1] for sub in subjects]
    train_labels = [sub.split(" ")[-1] for sub in subjects]

    subjects = open(val_list).read().splitlines()
    val_names = [sub.split(" ")[0].split('/')[-1] for sub in subjects]
    val_labels = [sub.split(" ")[-1] for sub in subjects]



    dic = {}
    dic_age = {}
    for i in range(len(train_names)):

        if train_labels[i] == "G":
            dic[train_names[i]] = 2.
        elif train_labels[i] == "A":
            dic[train_names[i]] = 0
        else:
            dic[train_names[i]] = 1.
        aa = train_all_df.loc[train_all_df['CPM_RadPath_2020_ID'] == train_names[i], ['age_in_days']].values[0][0]

        dic_age[train_names[i]] = aa / 365.0

    dic_val = {}
    dic_val_age = {}
    for i in range(len(val_names)):

        if val_labels[i] == "G":
            dic_val[val_names[i]] = 2.
        elif val_labels[i] == "A":
            dic_val[val_names[i]] = 0
        else:
            dic_val[val_names[i]] = 1.

        aa = train_all_df.loc[train_all_df['CPM_RadPath_2020_ID'] == val_names[i], ['age_in_days']].values[0][0]

        dic_val_age[val_names[i]] = aa / 365.0


    train_img = []
    train_label = []
    val_img = []
    val_label = []
    val_patch_count = []
    val_targets = []
    names = []
    train_age = []
    val_age=[]

    path = "./pathology_patches/"
    val_path = path

    random.seed(10)
    for name in train_names:

        if name in train_names:
            if dic[name] != 2:


                images = os.listdir(path + name)


                images = [path + name + "/" + img for img in images]

                train_img += images  # append(path+name+"/"+img)
                train_label += [dic[name]] * len(images)
                train_age += [dic_age[name]] * len(images)
    #
    for name in val_names:

        if dic_val[name] != 2:


            images = os.listdir(val_path + name)

            images = [val_path + name + "/" + img for img in images]

            train_img += images  # append(path+name+"/"+img)
            train_label += [dic_val[name]] * len(images)
            train_age += [dic_val_age[name]] * len(images)



    base_lr = 3e-4


    batch_size = config.batch_size
    config.model_name = config.model+'_'+str(config.image_size)+'_fold'+str(config.train_fold_index)+'_'+config.tag
    ## setup  -----------------------------------------------------------------------------
    out_dir = os.path.join(config.model_save_path, config.model_name)
    train_dataset = patchDataset_add_age(train_img, train_label,train_age, transform=1)

    train_loader  = DataLoader(train_dataset,
                                shuffle = True,
                                batch_size  = batch_size,
                                drop_last   = True,
                                num_workers = 12,
                                pin_memory  = True)

    valid_dataset = patchDataset_add_age(val_img, val_label,val_age,transform=None)

    valid_loader  = DataLoader(valid_dataset,
                                shuffle = False,
                                batch_size  = batch_size,
                                drop_last   = False,
                                num_workers = 12,
                                pin_memory  = True)

    net = get_model(config.model)


    ## optimiser ----------------------------------
    net = torch.nn.DataParallel(net)
    print(net)
    net = net.cuda()

    net.load_state_dict(torch.load('./pathology_G_best.pth'))  # load cls G best model to warm up




    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    lr_decay = torch.optim.lr_scheduler.MultiStepLR(optimizer, [50, 80])

    start_iter = 0

    i  = 0
    start = timer()

    loss_l1 = nn.SmoothL1Loss()  # nn.CrossEntropyLoss(ignore_index=-1)
    loss_ce = nn.CrossEntropyLoss()  # ocal_loss(alpha=0.25, gamma=2,num_classes = 3)# #nn.CrossEntropyLoss()


    loss_l1 = loss_l1.cuda()
    loss_ce = loss_ce.cuda()

    net.train()
    max_val_acc = 0.
    min_val_loss = 100.

    max_kappa = 0.0
    max_f1 = 0.0


    for epoch in range(config.train_epoch):

        lr_decay.step()


        log1.info('Start epoch {}'.format(epoch))


        Targets = []
        train_loss = 0
        net.train()

        optimizer.zero_grad()


        rate=lr_decay.get_lr()[0]

        # rate, hard_ratio = adjust_lr_and_hard_ratio(optimizer, epoch + 1)
        print('change lr: '+str(rate))


        for batch_id, batch_data in enumerate(train_loader):
            iter = i + start_iter
            # getting data batch

            volumes, label,age = batch_data
            Targets += list(label)
            label_ce = label.view(-1)
            label_ce[label_ce == 0] = 0



            volumes = volumes.cuda()
            label_l1 = label.type(torch.FloatTensor)
            label_l1 = label_l1.cuda()

            label_ce = label_ce.type(torch.LongTensor)
            label_ce = label_ce.cuda()

            age=age.cuda()


            optimizer.zero_grad()
            out_reg,out_cls= net(volumes,age)



            loss1 = loss_l1(out_reg, label_l1)
            loss2 = loss_ce(out_cls, label_ce)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(config.model_name + ' %0.7f %5.3f %6.3f | %0.3f %0.3f| %s' % (\
                             rate, iter, epoch,loss1.item(), loss2.item(),
                             time_to_str((timer() - start),'min')))

            i=i+1



        net.eval()

        val_loss = 0
        out_label1 = []
        out_label0 = []
        k = 0
        prediction0 = []
        prediction1 = []

        with torch.no_grad():

            for batch_id, batch_data in enumerate(valid_loader):
                # getting data batch

                volumes, label,age = batch_data

                label_ce = label.view(-1)
                label_ce[label_ce == 0] = 0



                volumes = volumes.cuda()

                label_l1 = label.type(torch.FloatTensor)
                label_l1 = label_l1.cuda()

                label_ce = label_ce.type(torch.LongTensor)
                label_ce = label_ce.cuda()
                age=age.cuda()


                out_reg,out_cls = net(volumes,age)

                loss1 = loss_l1(out_reg, label_l1)
                loss2 = loss_ce(out_cls, label_ce)
                loss = loss1 + loss2


                log1.info(
                    'Batch: {}-{}, loss1 = {:.3f}, loss2 = {:.3f}' \
                        .format(epoch, batch_id, loss1.item(), loss2.item()))

                val_loss += loss.detach().cpu().numpy()

                out_label0+=list(out_reg.detach().cpu().numpy())

                out1 = F.softmax(out_cls, dim=1)
                # print("y",out.shape,out_reg)
                out_label1 += list((out1.detach().cpu().numpy()))


                while len(out_label1) >= val_patch_count[k]:

                    out_label1_array = np.array(out_label1[:val_patch_count[k]])


                    pre1 = np.sum(out_label1_array, 0)
                    print(pre1.shape,out_label1_array.shape)
                    print(names[k], pre1 / val_patch_count[k])

                    prediction1.append(pre1 / val_patch_count[k])


                    out_label1 = out_label1[val_patch_count[k]:]

                    k = k + 1
                    if k >= len(val_patch_count):
                        break

        print("done")
        Targets = np.array(val_targets)
        Targets[Targets == 0] = 0



        prediction1 = np.array(prediction1)


        prediction1 = np.array(np.argmax(prediction1, axis=1))


        f1_score2 = metrics.f1_score(Targets[:], prediction1[:], average="micro")

        balan_acc2 = metrics.balanced_accuracy_score(Targets[:], prediction1[:])
        kappa2 = metrics.cohen_kappa_score(Targets[:], prediction1[:])
        acc2 = metrics.accuracy_score(Targets[:], prediction1[:])

        print("acc=", acc2, "Balan_acc:", balan_acc2, "kappa:",kappa2, "fiscore:", f1_score2)


        balan_acc = balan_acc2
        kappa = kappa2
        f1_score = f1_score2


        # # save model
        if balan_acc >= max_val_acc or val_loss <= min_val_loss or kappa >= max_kappa or f1_score >= max_f1:
            min_val_loss = val_loss
            max_val_acc = balan_acc
            max_kappa = kappa
            max_f1 = f1_score


        save_model_path = os.path.join(
            out_dir + '/epoch_{0}_val_loss_{1:.4f}_acc_{2:.4f}_kappa_{3:.4f}_fi_{4:.4f}.pth'.format(
                epoch, val_loss / (batch_id + 1), balan_acc, kappa, f1_score))

        torch.save(net.state_dict(), save_model_path)





def main(config):

    if config.mode == 'train':
        run_train(config)



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='cpm2020')
    parser.add_argument('--train_fold_index', type=int, default = 4)
    parser.add_argument('--model', type=str, default='G_cls')
    parser.add_argument('--batch_size', type=int, default=96)


    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--model_save_path', type=int, default=2)
    parser.add_argument('--train_epoch', type=int, default=100)
    config = parser.parse_args()
    print(config)
    main(config)



