import numpy as np
import random
import torchvision
import torch

# Dataset iterator
def inf_train_gen(n_samples):
  scale = 3
  num_samples = n_samples
 
  centers = [
      (0, 0, 0),
      (1, 1, 0),
      (1, -1, 0),
      (1, 0, 1),
      (1 , 0, -1),
      #(-1, 1, 0),
      #(-1, -1, 0 ),
      #(-1, 0, 1),
      #(-1, 0, -1),
  ]
  centers = [(scale * x, scale * y, scale * z ) for x, y, z in centers]
  #while True:
  dataset = []
  for i in range(num_samples):
      point = np.random.randn(3) * 0.2
      point[2] = point[2]*random.choice([1,-1])
      center = random.choice(centers)
      point[0] += center[0]
      point[1] += center[1]
      point[2] += center[2]
      dataset.append(point)
  dataset = np.array(dataset, dtype='float32')
  dataset /= 1.414 # stdev
  return dataset


def mnist_loader():
  #Converting data to torch.FloatTensor
  transform = torchvision.transforms.ToTensor()

  # Download the training and test datasets
  train_data = torchvision.datasets.MNIST(root='data', train=True, download=True, transform=transform)

  test_data = torchvision.datasets.MNIST(root='data', train=False, download=True, transform=transform)

  train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000, num_workers=0)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000, num_workers=0)
  print(len(test_data))
  print(len(train_data))

  classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

  # dataiter = iter(train_loader)
  # images, labels = dataiter.next()
  # images = images.numpy()

  return train_data, test_data, classes
