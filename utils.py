from tqdm.auto import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt



def train_step(model,
              dataloader:torch.utils.data.DataLoader,
              loss_fn,
              optimizer,
              device):
    model.to(device)
    model.train()
    train_loss,train_acc=0,0
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)
        
        y_pred=model(X)
        
        loss=loss_fn(y_pred,y)
        
        train_loss+=loss.item()
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
        
        y_pred_class=torch.softmax(y_pred,dim=1).argmax(dim=1)
        train_acc += (y_pred_class==y).sum().item()/len(y_pred)

    
    train_loss/=len(dataloader)
    train_acc/=len(dataloader)
    
    return train_loss,train_acc



        
        
        
def test_step(model,
             dataloader:torch.utils.data.DataLoader,
             loss_fn,
             optimizer,
             device):
    model.to(device)
    model.eval()
    test_loss,test_acc=0,0
    
    with torch.inference_mode():
        for batch,(X,y) in enumerate(dataloader):
            
            X,y=X.to(device),y.to(device)
            
            test_pred_logits=model(X)
            
            loss=loss_fn(test_pred_logits,y)
            
            test_loss+=loss.item()
            
            test_pred_labels=torch.softmax(test_pred_logits,dim=1).argmax(dim=1)
            
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
            
    test_loss/=len(dataloader)
    test_acc/=len(dataloader)
            
    return test_loss,test_acc



def train(model,
         train_dataloader:torch.utils.data.DataLoader,
         test_dataloader:torch.utils.data.DataLoader,
         optimizer,
         loss_fn,
         device,
         epochs):
    results={"train_loss":[],
             "train_acc":[],
             "test_loss":[],
             "test_acc":[]}
    
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc=train_step(model=model,
                                       dataloader=train_dataloader,
                                       loss_fn=loss_fn,
                                       optimizer=optimizer,
                                       device=device)
        test_loss,test_acc=test_step(model=model,
                                     dataloader=test_dataloader,
                                     loss_fn=loss_fn,
                                     optimizer=optimizer,
                                     device=device)
        print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
        
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
    return results


def plot_loss_curves(results):
    loss=results['train_loss']
    test_loss=results['test_loss']
    
    acc=results['train_acc']
    test_acc=results['test_acc']
    
    epochs=range(len(results['train_loss']))
    
    plt.figure(figsize=(15,7))
    
    plt.subplot(1,2,1)
    plt.plot(epochs,loss,label='train_loss')
    plt.plot(epochs,test_loss,label='test_loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    
    
    plt.subplot(1,2,2)
    plt.plot(epochs,acc,label='train_acc')
    plt.plot(epochs,test_acc,label='test_acc')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend();
    
    
    