
import random
import numpy as np
from data import get_cityscapes, get_camvid
from model_cross_dataset import RegSeg
from losses import *
from lr_schedulers import *
import torch.cuda.amp as amp

class ConfusionMatrix(object):
    def __init__(self, num_classes, exclude_classes):
        self.num_classes = num_classes
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        self.exclude_classes=exclude_classes

    def update(self, a, b):
        a=a.cpu()
        b=b.cpu()
        n = self.num_classes
        k = (a >= 0) & (a < n)
        inds = n * a + b
        inds=inds[k]
        self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))

        acc_global=acc_global.item() * 100
        acc=(acc * 100).tolist()
        iu=(iu * 100).tolist()
        return acc_global, acc, iu
    def __str__(self):
        acc_global, acc, iu = self.compute()
        acc_global=round(acc_global,2)
        IOU=[round(i,2) for i in iu]
        mIOU=sum(iu)/len(iu)
        mIOU=round(mIOU,2)
        reduced_iu=[iu[i] for i in range(self.num_classes) if i not in self.exclude_classes]
        mIOU_reduced=sum(reduced_iu)/len(reduced_iu)
        mIOU_reduced=round(mIOU_reduced,2)
        return f"IOU: {IOU}\nmIOU: {mIOU}, mIOU_reduced: {mIOU_reduced}, accuracy: {acc_global}"

def evaluate(model, data_loader, device, confmat,mixed_precision,print_every,max_eval,dataname):
    model.eval()
    assert(isinstance(confmat,ConfusionMatrix))
    with torch.no_grad():
        for i,(image, target) in enumerate(data_loader):
            if (i+1)%print_every==0:
                print(i+1)
            image, target = image.to(device), target.to(device)
            with amp.autocast(enabled=mixed_precision):
                output = model(image,dataname)
            output = torch.nn.functional.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
            confmat.update(target.flatten(), output.argmax(1).flatten())
            if i+1==max_eval:
                break
    return confmat

def setup_env():
    torch.backends.cudnn.benchmark=True
    seed=0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) 

def train():
    train_loader_city, val_loader_city, train_set_city = get_cityscapes(
        root = "/home/stua/cl/CLReg_prog/cityscapes_dataset/" ,  
        batch_size = 8,  
        train_min_size = 400,  
        train_max_size = 1600,  
        train_crop_size = [768,768], 
        val_input_size = 1024,
        val_label_size = 1024,
        aug_mode = "randaug_reduced",
        class_uniform_pct = 0.5,
        train_split = "train",
        val_split = "val",
        num_workers = 6,
        ignore_value = 255
    )
    train_loader_cam, val_loader_cam, train_set_cam = get_camvid(
        root = "./camvid_dataset",
        batch_size=2,
        train_min_size=288,
        train_max_size=1152,
        train_crop_size= [720,960],
        val_input_size=720,
        val_label_size=720,
        aug_mode = "baseline",
        train_split = "trainval",
        val_split = "test",
        num_workers = 4,
        ignore_value = 255
    )
    
    setup_env()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    save_best_path = "./pth_file/crossdataset_test0"
    print("saving to: " + save_best_path)
    max_epochs = 500
    epochs = 500
    mixed_precision  =True
    log_path = "./log_file/crossdataset_test0"


    model = RegSeg()

    total_iterations = min(len(train_loader_city),len(train_loader_cam))*max_epochs
    optim_params = model.parameters()
    optimizer = torch.optim.SGD(optim_params,lr=0.01,momentum=0.9,weight_decay=0.0001)
    scaler = amp.GradScaler(enabled=mixed_precision)
    loss_fun = OhemCrossEntropy2d(thresh=0.7,n_min=100000,ignore_lb=255)
    lr_function = lambda x : poly_lr_scheduler(x,total_iterations,4000,0.1,0.9)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_function)

    epoch_start = 0

    with open(log_path, "a") as f:
        f.write(f"Begin Trainning!!!!!!\n")
    for epoch in range(epoch_start, epochs):
        torch.manual_seed(epoch)
        random.seed(epoch)
        np.random.seed(epoch)

        model.train()
        model.to(device)
        # model.cpu()
        losses = 0
        torch.set_printoptions(threshold=np.inf)

        for x_city, x_cam in zip(train_loader_city, train_loader_cam):  
            image_city, label_city = x_city
            image_city, label_city = image_city.cuda(), label_city.cuda()
            image_cam, label_cam = x_cam
            image_cam, label_cam = image_cam.cuda(), label_cam.cuda()
            with amp.autocast(enabled=mixed_precision):
                output1 = model(image_city, "Cityscapes")
                loss1 = loss_fun(output1, label_city)
                # if epoch < 400:
                #     loss2 = 0
                # else:
                output2 = model(image_cam, "CamVid")
                loss2 = loss_fun(output2, label_cam)
                loss = loss1 + loss2
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            losses += loss.item()
        print(losses/int(min(len(train_loader_city),len(train_loader_cam))))
        with open(log_path, "a") as f:
            f.write(f"{losses/int(min(len(train_loader_city),len(train_loader_cam)))}\n")
        if epoch%25==0 and epoch<470 :
            confmat = ConfusionMatrix(19,[])
            confmat = evaluate(model,val_loader_city,device,confmat,mixed_precision,1000,500,"Cityscapes")
            with open(log_path, "a") as f:
                f.write(f"Cityscapes: ")
                f.write(f"{confmat}\n")
            print(confmat)
            confmat2 = ConfusionMatrix(12, [])
            confmat2 = evaluate(model, val_loader_cam, device, confmat2, mixed_precision, 1000, 500, "CamVid")
            with open(log_path, "a") as f:
                f.write(f"CamVid: ")
                f.write(f"{confmat2}\n")
            print(confmat2)
        elif epoch >= 470 :
            confmat = ConfusionMatrix(19, [])
            confmat = evaluate(model, val_loader_city, device, confmat, mixed_precision, 1000, 500, "Cityscapes")
            with open(log_path, "a") as f:
                f.write(f"Cityscapes: ")
                f.write(f"{confmat}\n")
            print(confmat)
            confmat2 = ConfusionMatrix(12, [])
            confmat2 = evaluate(model, val_loader_cam, device, confmat2, mixed_precision, 1000, 500, "CamVid")
            with open(log_path, "a") as f:
                f.write(f"CamVid: ")
                f.write(f"{confmat2}\n")
            print(confmat2)

    
if __name__ == "__main__":
    torch.cuda.set_device(3)
    train()
