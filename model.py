import torch
from torch import nn

class TinyVGG(nn.Module):
    def __init__(self,input_features,hidden_features,output_features):
        super().__init__()
        self.conv_block_1=nn.Sequential(nn.Conv2d(in_channels=input_features,
                                                  out_channels=hidden_features,
                                                  kernel_size=2,
                                                  stride=1,
                                                  padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=hidden_features,
                                                  out_channels=hidden_features,
                                                  kernel_size=2,
                                                  stride=1,
                                                  padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=2)
                                       )
        self.conv_block_2=nn.Sequential(nn.Conv2d(in_channels=hidden_features,
                                                 out_channels=hidden_features,
                                                 kernel_size=2,
                                                 padding=1,
                                                 stride=1),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=hidden_features,
                                                 out_channels=hidden_features,
                                                 kernel_size=2,
                                                 stride=1,
                                                 padding=1),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2)
                                        
                                       
                                      )
        self.conv_block_3=nn.Sequential(nn.Conv2d(in_channels=hidden_features,
                                         out_channels=hidden_features,
                                         kernel_size=2,
                                         padding=1,
                                         stride=1),
                               nn.ReLU(),
                               nn.Conv2d(in_channels=hidden_features,
                                         out_channels=hidden_features,
                                         kernel_size=2,
                                         stride=1,
                                         padding=1),
                               nn.ReLU(),
                               nn.MaxPool2d(kernel_size=2))


        self.classifier=nn.Sequential(nn.Flatten(),
                                      nn.Linear(in_features=hidden_features*9*9,
                                                out_features=output_features))
        
    def forward(self,x):
        x=self.conv_block_1(x)
#         print(x.shape)
        x=self.conv_block_2(x)
#         print(x.shape)
        x=self.conv_block_3(x)
#         print(x.shape)

#         print(x.shape)
        x=self.classifier(x)
        return x
        