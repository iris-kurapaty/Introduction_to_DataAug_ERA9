from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm

class Block(nn.Module):
  def __init__(self, input_size, out_size, drop_out):
    super(Block, self).__init__()
    self.drop_out = drop_out

    self.convblock1 = nn.Sequential(
        nn.Conv2d(in_channels=input_size, out_channels=out_size, kernel_size=(3,3), padding=1, bias=False),
        nn.BatchNorm2d(out_size),
        nn.Dropout(drop_out),
        nn.ReLU()
    )# output_size = 32; RF = 3

    self.convblock2 = nn.Sequential(
        nn.Conv2d(in_channels=out_size, out_channels = out_size, kernel_size=(3,3), padding=1, bias=False),
        nn.BatchNorm2d(out_size),
        nn.Dropout(drop_out),
        nn.ReLU()
    )# output_size = 32; RF = 5

    self.convblock3 = nn.Sequential(
        nn.Conv2d(out_size, out_size, kernel_size = (3,3), padding=1, dilation = 2, stride=2, bias=False),
        nn.BatchNorm2d(out_size),
        nn.Dropout(drop_out),
        nn.ReLU()
    )# output_size = 32; RF = 9

  def __call__(self, x):
    x = self.convblock1(x)
    x = self.convblock2(x) 
    x = self.convblock3(x) 
    
    return x

class DepthWiseConvolution(nn.Module):
  def __init__(self, input_size, output_size):
    super(DepthWiseConvolution, self).__init__()

    self.depthwise1 = nn.Sequential(
        nn.Conv2d(input_size, input_size, kernel_size = (3,3),padding= 1,groups = input_size),
        nn.ReLU()
    )
    self.pointwise1 =  nn.Sequential(
        nn.Conv2d(input_size, output_size, kernel_size = (1,1)),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )
    self.depthwise2 = nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (3,3),padding= 1,groups = output_size),
        nn.ReLU()
    )
    self.pointwise2 =  nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (1,1)),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )
    self.depthwise3 = nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (3,3),padding= 1,groups = output_size),
        nn.ReLU()
    )
    
    self.pointwise3 =  nn.Sequential(
        nn.Conv2d(output_size, output_size, kernel_size = (1,1), padding= 0),
        nn.BatchNorm2d(output_size),
        nn.ReLU()
    )
   

  def __call__(self, x):
    x = self.depthwise1(x)
    x = self.pointwise1(x)
    x = self.depthwise2(x)
    x = self.pointwise2(x)    
    x = self.depthwise3(x)
    x = self.pointwise3(x)
    return x

# Block 1: 3, 5, 9    
# Block 2: 13, 17, 25
# Block 3: 25, 33, 41
# Block 4: 49, 57, 65

class Net(nn.Module):
  def __init__(self, drop_out = 0.1):
    super(Net, self).__init__()
    self.drop_out = drop_out

    # Input Block + Convolution Blocks
    self.layer1 = Block(3, 32, 0.1)
    self.layer2 = Block(32, 64, 0.1)

    # Depth-Wise Separable Convolutions
    self.layer3 = DepthWiseConvolution(64, 128)

   # OUTPUT BLOCK

    # output_size = 4; ; RF = 50
    self.gap = nn.Sequential(
        nn.AvgPool2d(kernel_size=7)
    ) # output_size = 1

    self.convblock5 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
    )# output_size = 1; ; RF = 28

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.gap(x)
    x = self.convblock5(x)
    x = x.view(-1, 10)
    return F.log_softmax(x, dim=-1)
    
def train(model, device, train_loader, optimizer, epoch):
  train_losses = []
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc = (100*correct/processed)

  return train_losses, train_acc


def test(model, device, test_loader):
  test_losses = []
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
      pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc = (100. * correct / len(test_loader.dataset))
  return test_losses, test_acc

  