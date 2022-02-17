import nibabel as nib
from setting import parse_opts
from datasets.brain import BrainDataset_AO,BrainDataset_G

from model import generate_model
from loss import focal_loss
from models import densenet
import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import time
from utils.logger import log
from scipy import ndimage
from sklearn import metrics
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def train(data_loader, val_data_loader,model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # settings
    batches_per_epoch = len(data_loader)
    val_batches_per_epoch=len(val_data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_l1 = nn.SmoothL1Loss()#nn.CrossEntropyLoss(ignore_index=-1)
    loss_ce =nn.CrossEntropyLoss()#ocal_loss(alpha=0.25, gamma=2,num_classes = 3)# #nn.CrossEntropyLoss()
    print("Current setting is:")
    print(sets)
    print("\n\n")     
    if not sets.no_cuda:
        loss_l1 = loss_l1.cuda()
        loss_ce = loss_ce.cuda()
        
    model.train()
    train_time_sp = time.time()
    max_val_acc = 0.
    min_val_loss = 100.

    max_kappa = 0.0
    max_f1 = 0.0
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        log.info('lr = {}'.format(scheduler.get_lr()))
        Targets = []
        out1 = []
        out2 = []
        train_loss=0
        model.train()
        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch

            volumes, labels = batch_data

            Targets += list(labels)

            label_ce=labels.view(-1)
            label_ce[label_ce==0.5]=0

            #volumes = volumes[:, :4]

            if not sets.no_cuda:
                volumes = volumes.cuda()
                new_labels=labels.cuda()
                label_ce = label_ce.type(torch.LongTensor)
                label_ce=label_ce.cuda()





            optimizer.zero_grad()
            out = model(volumes)




            # calculating loss
            loss1 = loss_l1(out[0], new_labels)
            loss2 = loss_ce(out[1],label_ce)
            loss = loss1+loss2
            loss.backward()
            optimizer.step()
            out1+=list(out[0].detach().cpu().numpy())


            out2+= list(torch.max(out[1], 1)[1].detach().cpu().numpy()) #

            log.info(
                    'Batch: {}-{}, loss1 = {:.3f}, loss2 = {:.3f}'\
                    .format(epoch, batch_id, loss1.item(), loss2.item()))

            train_loss+=loss.detach().cpu().numpy()

        scheduler.step()

        Targets = np.array(Targets)
        Targets[Targets == 0.5] = 0

        out1 = np.array(out1)
        out1=out1.reshape(-1)
        prediction = np.zeros(out1.shape)
        thrs = [0.5]
        # print(outputs)
        prediction[out1 < thrs[0]] = 0.
        prediction[(out1 >= thrs[0])]=1


        prediction2 = np.array(out2)

        f1_score = metrics.f1_score(Targets, prediction, average="micro")
        balan_acc = metrics.balanced_accuracy_score(Targets, prediction)
        kappa = metrics.cohen_kappa_score(Targets, prediction)
        acc=metrics.accuracy_score(Targets,prediction)
        print("epoch:", epoch, "train_loss:", train_loss/(batch_id+1))
        print("acc=",acc,"Balan_acc:", balan_acc, "kappa:", kappa, "fiscore:",f1_score)

        f1_score = metrics.f1_score(Targets, prediction2, average="micro")
        balan_acc = metrics.balanced_accuracy_score(Targets, prediction2)
        kappa = metrics.cohen_kappa_score(Targets, prediction2)
        acc = metrics.accuracy_score(Targets, prediction2)
        print( "acc=", acc, "Balan_acc:", balan_acc,"kappa:", kappa, "fiscore:", f1_score)


        model.eval()

        val_loss=0
        Targets = []
        out1 = []
        out2=[]
        for batch_id, batch_data in enumerate(val_data_loader):
            # getting data batch

            volumes, labels = batch_data


            Targets += list(labels)

            label_ce=labels.view(-1)
            label_ce[label_ce==0.5]=0



            if not sets.no_cuda:
                volumes = volumes.cuda()
                new_labels = labels.cuda()
                label_ce = label_ce.type(torch.LongTensor)
                label_ce=label_ce.cuda()

            out = model(volumes)

            loss1 = loss_l1(out[0], new_labels)
            loss2 = loss_ce(out[1],label_ce)
            loss = loss1+loss2

            out1 += list(out[0].detach().cpu().numpy())

            # print(out_masks.view(-1).detach().cpu())
            out2 += list(torch.max(out[1], 1)[1].detach().cpu().numpy())  # 1返回index  0返回原值

            log.info(
                'Batch: {}-{}, loss1 = {:.3f}, loss2 = {:.3f}' \
                    .format(epoch, batch_id, loss1.item(), loss2.item()))

            val_loss += loss.detach().cpu().numpy()

        Targets = np.array(Targets)
        Targets[Targets == 0.5] = 0


        out1 = np.array(out1)
        out1 = out1.reshape(-1)
        prediction = np.zeros(out1.shape)
        thrs = [0.5]
        # print(outputs)
        prediction[out1 < thrs[0]] = 0.
        prediction[(out1 >= thrs[0])]=1#


        prediction2 = np.array(out2)


        f1_score1 = metrics.f1_score(Targets[:], prediction[:], average="micro")
        balan_acc1 = metrics.balanced_accuracy_score(Targets[:], prediction[:])
        kappa1 = metrics.cohen_kappa_score(Targets[:], prediction[:])
        acc = metrics.accuracy_score(Targets[:], prediction[:])
        print("epoch:", epoch, "val_loss:", val_loss / (batch_id + 1))
        print("l1","acc=", acc, "Balan_acc:", balan_acc1, "kappa:", kappa1, "fiscore:", f1_score1)

        f1_score2 = metrics.f1_score(Targets[:], prediction2[:], average="micro")
        balan_acc2 = metrics.balanced_accuracy_score(Targets[:], prediction2[:])
        kappa2 = metrics.cohen_kappa_score(Targets[:], prediction2[:])
        acc = metrics.accuracy_score(Targets[:], prediction2[:])
        print("ce","acc=", acc, "Balan_acc:", balan_acc2, "kappa:", kappa2, "fiscore:", f1_score2)



        f1_score1_valid = metrics.f1_score(Targets[:], prediction[:], average="micro")
        balan_acc1_valid = metrics.balanced_accuracy_score(Targets[:], prediction[:])
        kappa1_valid = metrics.cohen_kappa_score(Targets[:], prediction[:])
        acc = metrics.accuracy_score(Targets[:], prediction[:])
        print("valid_l1", "acc=", acc, "Balan_acc:", balan_acc1, "kappa:", kappa1, "fiscore:", f1_score1)

        f1_score2_valid = metrics.f1_score(Targets[:], prediction2[:], average="micro")
        balan_acc2_valid = metrics.balanced_accuracy_score(Targets[:], prediction2[:])
        kappa2_valid = metrics.cohen_kappa_score(Targets[:], prediction2[:])
        acc = metrics.accuracy_score(Targets[:], prediction2[:])
        print("valid_ce", "acc=", acc, "Balan_acc:", balan_acc2, "kappa:", kappa2, "fiscore:", f1_score2)


        if balan_acc1+kappa1+f1_score1>balan_acc2+kappa2+f1_score2:
            balan_acc=balan_acc1
            kappa=kappa1
            f1_score=f1_score1
        else:
            balan_acc=balan_acc2
            kappa=kappa2
            f1_score=f1_score2

        if balan_acc1_valid + kappa1_valid + f1_score1_valid > balan_acc2_valid + kappa2_valid + f1_score2_valid:
            balan_acc_valid = balan_acc1_valid
            kappa_valid = kappa1_valid
            f1_score_valid = f1_score1_valid

        else:
            balan_acc_valid = balan_acc2_valid
            kappa_valid = kappa2_valid
            f1_score_valid = f1_score2_valid


        save_model_path = os.path.join(save_folder+'epoch_{0}_val_loss_{1:.4f}_acc_{2:.4f}_kappa_{3:.4f}_fi_{4:.4f}_valid_acc_{5:.4f}_kappa_{6:.4f}_fi_{7:.4f}.pth'.format(
                                           epoch, val_loss/(batch_id+1), balan_acc, kappa, f1_score,balan_acc_valid,kappa_valid,f1_score_valid))

        torch.save(model.state_dict(), save_model_path)

                            
    print('Finished training')            



if __name__ == '__main__':
    # settting
    sets = parse_opts()   


    # getting model
    torch.manual_seed(sets.manual_seed)
    RESNET=False
    if RESNET:
        model = generate_model(sets)
        net_dict = model.state_dict()
        #
        print('loading pretrained model {}'.format(sets.pretrain_path))
        pretrain = torch.load(sets.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        model.conv1 = nn.Conv3d(
            5,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
    else:
        model = densenet.densenet121(num_classes=2,
                                     sample_size=192, first=5,
                                     sample_duration=128,
                                     drop_rate=0.2)

    # # # #
    parameters = model.parameters()

    model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0,1])
    # model = model.module


    optimizer = torch.optim.Adam(params=parameters, lr=sets.learning_rate)#,momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    model.load_state_dict(torch.load('./G_best.pth'))  #load cls G best model to warm up

    print (model)



    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True

    training_dataset = BrainDataset_AO(sets.train_img_list, sets,"train")
    data_loader = DataLoader(training_dataset, batch_size=16, shuffle=True, num_workers=12, pin_memory=False)#sets.pin_memory)# 16  24

    val_dataset = BrainDataset_AO(sets.val_img_list, sets,"valid")
    val_data_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4,pin_memory=False)#sets.pin_memory) # 4    4


    # training
    train(data_loader,val_data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets)
