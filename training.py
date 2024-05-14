import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from LYTNET import LYTNet
from LYTNETV2 import LYTNetV2
from dataset import TrafficLightDataset
from loss import my_loss
from helpers import direction_performance

cuda_available = torch.cuda.is_available()

BATCH_SIZE = 32
MAX_EPOCHS = 800
INIT_LR = 0.001
WEIGHT_DECAY = 0.00005
LR_DROP_MILESTONES = [400,600]

train_file_dir = '/Users/yuvvan_talreja/Desktop/Coding/CrossyRoad/training_file.csv'
valid_file_dir = '/Users/yuvvan_talreja/Desktop/Coding/CrossyRoad/validation_file.csv'
train_img_dir = '/Users/yuvvan_talreja/Desktop/Coding/CrossyRoad/Crossy_Dataset_876x657'
valid_img_dir = '/Users/yuvvan_talreja/Desktop/Coding/CrossyRoad/Crossy_Dataset_768x576'
save_path = '/Users/yuvvan_talreja/Desktop/Coding/CrossyRoad/'

train_dataset = TrafficLightDataset(csv_file = train_file_dir, img_dir = train_img_dir)
valid_dataset = TrafficLightDataset(csv_file = valid_file_dir, img_dir = valid_img_dir)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

net = LYTNet()

if cuda_available:
    net = net.cuda()

loss_fn = my_loss

optimizer = torch.optim.Adam(net.parameters(), lr = INIT_LR, weight_decay = 0.000005 )
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, LR_DROP_MILESTONES)

#storing all data during training
train_losses = [] #stores the overall training loss at each epoch
train_MSE = [] #stores the MSE loss during training at each epoch
train_CE = [] #stores the cross entropy loss during training at each epoch
valid_losses = [] #stores the overall validation loss at each epoch
valid_MSE = [] #stores the MSE loss during validation at each epoch
valid_CE = [] #stores the cross entropy loss during validation at each epoch
train_accuracies = [] #stores the training accuracy of the network at each epoch
valid_accuracies = [] #stores the validation accuracy of the network at each epoch
val_angles = [] #stores the average angle error of the network during validation at each epoch
val_start = [] #stores the average startpoint error of the network during validation at each epoch
val_end = [] #stores the average endpoint error of the network during validation at each epoch

if __name__ == '__main__':

    for epoch in range(MAX_EPOCHS):
        
        
        net.train()
        
        running_loss = 0.0 #stores the total loss for the epoch
        running_loss_MSE = 0.0 #stores the total MSE loss for the epoch
        running_loss_cross_entropy = 0.0 #store the total cross entropy loss for the epoch
        angle_error = 0.0 #stores average angle error for the epoch
        startpoint_error = 0.0 #stores average startpoint error for the epoch
        endpoint_error = 0.0 #stores average endpoint error for the epoch
        train_correct = 0 #stores total number of correctly predicted images during training for the epcoh
        train_total = 0 #stores total number of batches processed at each epoch
        
        for j, data in enumerate(train_dataloader, 0): 
            
            optimizer.zero_grad()
            train_total += 1
            
            images = data['image'].type(torch.FloatTensor)
            mode = data['mode'] #index of traffic light mode
            points = data['points'] #array of midline coordinates
            
            if cuda_available:
                images = images.cuda()
                mode = mode.cuda()
                points = points.cuda()
            
            pred_classes, pred_direc = net(images)
            _, predicted = torch.max(pred_classes, 1) #finds index of largest probability
            train_correct += (predicted == mode).sum().item() #increments train_correct if predicted index is correct
            loss, MSE, cross_entropy = loss_fn(pred_classes, pred_direc, points, mode)
            angle, start, end = direction_performance(pred_direc, points)
            angle_error += angle
            endpoint_error += end
            startpoint_error += start
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_loss_MSE += MSE.item()
            running_loss_cross_entropy += cross_entropy.item()

        print('Epoch: ' + str(epoch+1))
        print('Average training loss: ' + str(running_loss/(j+1)))
        print('Average training MSE loss: ' + str(running_loss_MSE/(j+1)))
        print('Average training cross entropy loss: ' + str(running_loss_cross_entropy/(j+1)))
        print('Training accuracy: ' + str(train_correct/train_total/BATCH_SIZE))

        train_MSE.append(running_loss_MSE/train_total)
        train_CE.append(running_loss_cross_entropy/train_total)
        train_losses.append(running_loss/train_total) 
        train_accuracies.append(train_correct/train_total/32*100) 
                
        lr_scheduler.step(epoch + 1) #decrease learning rate if at desired epoch   
        