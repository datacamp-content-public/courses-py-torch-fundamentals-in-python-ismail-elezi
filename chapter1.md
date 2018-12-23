---
title: 'Artificial Neural Networks'
description: 'Chapter description goes here.'
---

## Activation Functions

```yaml
type: VideoExercise
key: 89911de4c7
xp: 50
```

`@projector_key`
83c000a6e8fa06235ab589e6835e9684

---

## Neural Networks

```yaml
type: NormalExercise
key: ba27197060
xp: 100
```

Let's see the differences between neural networks which apply `ReLU` and those which do not apply `ReLU`. We have given the code here which randomly initializes the input called `input_layer`, and three sets of weights, called `weight_1`, `weight_2` and `weight_3`.

```
input_layer = torch.randn(1, 4)
weight_1 = torch.randn(4, 4)
weight_2 = torch.randn(4, 4)
weight_3 = torch.randn(4, 4)
```

We are going to convince ourselves that networks with multiple layers which do not contain non-linearity can be expressed as neural networks with one layer.

`@instructions`
1) Multiply the `input_layer` with `weight_1` and store the result in `hidden_1`, then multiply `hidden_1` with `weight_2` and store the result in `hidden_2`.

2) Multiply `hidden_2` with `weight_3` and print the result.

3) Multiply `weights_1` with `weights_2` and store the result in `weight_composed_1`, then multiply `weight_composed_1` with `weight_3` and store the result in `weight`.

4) Multiply `input_layer` with `weight` and print the results.

`@hint`


`@pre_exercise_code`
```{python}
input_layer = torch.randn(1, 4)
weight_1 = torch.randn(4, 4)
weight_2 = torch.randn(4, 4)
weight_3 = torch.randn(4, 4)
```

`@sample_code`
```{python}
import torch
import torch.nn as nn

hidden_1 = torch.matmul(___, ___)
hidden_2 = torch.matmul(___, ___)

print(torch.matmul(___, ___))

weight_composed_1 = torch.matmul(___, ___)
weight = torch.matmul(___, ___)

print(torch.matmul(___, ___))
```

`@solution`
```{python}
import torch
import torch.nn as nn

hidden_1 = torch.matmul(input_layer, weight_1)
hidden_2 = torch.matmul(hidden_1, weight_2)

print(torch.matmul(hidden_2, weight_3))

weight_composed_1 = torch.matmul(weight_1, weight_2)
weight = torch.matmul(weight_composed_1, weight_3)

print(torch.matmul(input_layer, weight))
```

`@sct`
```{python}

```

---

## ReLU activation

```yaml
type: NormalExercise
key: 2ce188a22b
xp: 100
```

We have the same input and same weights as before. Now we are going to build a neural network which has non-linearity and by doing so, we are going to convince ourselves that networks with multiple layers and non-linearity functions cannot be expressed as neural networks with one layer.

`@instructions`
1) We have here the code from the previous exercise. Apply non-linearity on `hidden_1` and `hidden_2` calling those variables `hidden_1_activated` and `hidden_2_activated`.

2) Multiply `hidden_2_activated` with `weight_3` and print the result.

3) Apply non-linearity in `weight_composed_1` and store the result in `weight_composed_1_activated`, then multiply `weight_composed_1_activated` with `weight_3` and store the result in `weight`.

4) Multiply `input_layer` with `weight` and print the results.

`@hint`


`@pre_exercise_code`
```{python}

```

`@sample_code`
```{python}
import torch
import torch.nn as nn
relu = nn.ReLU()

hidden_1 = torch.matmul(input_layer, weight_1)
hidden_1_activated = ___(hidden_1)
hidden_2 = torch.matmul(hidden_1, weight_2)
hidden_2_activated = ___
print(torch.matmul(___, ___))

weight_composed_1 = torch.matmul(weight_1, weight_2)
weight_composed_1_activated = ___(weight_composed_1)
weight = torch.matmul(___, ___)
print(torch.matmul(___, ___))
```

`@solution`
```{python}
import torch
import torch.nn as nn
relu = nn.ReLU()

hidden_1 = torch.matmul(input_layer, weight_1)
hidden_1_activated = relu(hidden_1)
hidden_2 = torch.matmul(hidden_1, weight_2)
hidden_2_activated = relu(hidden_2)
print(torch.matmul(hidden_2, weight_3))

weight_composed_1 = torch.matmul(weight_1, weight_2)
weight_composed_1_activated = relu(weight_composed_1)
weight = torch.matmul(weight_composed_1_activated, weight_3)
print(torch.matmul(input_layer, weight))
```

`@sct`
```{python}

```

---

## Leaky ReLU

```yaml
type: NormalExercise
key: b0fd004979
xp: 100
```

On the previous exercise, it is possible that you got a vector of zeros in the output. The reason for this is that if a unit is connected with the previous layer only with negative weights, then that unit will become 0 after applying `ReLU` on it. Then that unit won't have any effect on the next layer, and so on. When a unit has value 0, that unit is considered to be dead. Based on some studies, in modern neural networks circa 30% of units are dead units.

In order to fix this, there is a simple trick. You can replace `ReLU` with `leaky ReLU` which implements the function `leaky_ReLU(x) = max(a*x, x)` where `a` is a small positive number, meaning that negative values instead of being set to 0 are set to a small number instead (the number being `a` times original value).

`@instructions`
The code is the same as in the previous exercise, but now we are going to replace `ReLU` with `leaky_ReLU`, and for parameter `a`, we are going to choose `0.1`.

1) Instantiate an object of class `nn.LeakyReLU` with `a = 0.1`. Hint: See how we instantiated `ReLU` before and replace `nn.ReLU` with `nn.LeakyReLU`.

2) Apply non-linearity on `hidden_1` and `hidden_2` calling those variables `hidden_1_activated` and `hidden_2_activated`.

3) Multiply `hidden_2_activated` with `weight_3` and print the result.

`@hint`


`@pre_exercise_code`
```{python}
input_layer = torch.randn(1, 4)
weight_1 = torch.randn(4, 4)
weight_2 = torch.randn(4, 4)
weight_3 = torch.randn(4, 4)
```

`@sample_code`
```{python}
import torch
import torch.nn as nn

leaky_ReLU = ___(0.1)

hidden_1 = torch.matmul(input_layer, weight_1)
hidden_1_activated = ___(hidden_1)
hidden_2 = torch.matmul(hidden_1, weight_2)
hidden_2_activated = ___
print(torch.matmul(___, ___))
```

`@solution`
```{python}
import torch
import torch.nn as nn

leaky_ReLU = nn.LeakyReLU(0.1)

hidden_1 = torch.matmul(input_layer, weight_1)
hidden_1_activated = leaky_ReLU(hidden_1)
hidden_2 = torch.matmul(hidden_1, weight_2)
hidden_2_activated = leaky_ReLU(hidden_2)
print(torch.matmul(hidden_2_activated, weight_3))
```

`@sct`
```{python}

```

---

## Softmax activation

```yaml
type: NormalExercise
key: 6e5a125d55
xp: 100
```

On the previous exercises, we did not insert non-linearity on the final layer. On neural networks, the final layer typically has a different version of non-linearity, whose purpose (in addition to adding non-linearity) is to transform the output values into a probability distribution. This allows the network to also point out the confidence of its prediction.

This type of function is called `softmax` function (`logistic` or `sigmoid` function on binary case) and is implemented on `torch.nn` module. Let us see how it works.

`@instructions`
1) Create three torch random tensors. The first one is called `input_1` and has shape 4 by 4, the second one is called `input_2` and has shape 1 by 4, the third one is called `input_3` and has shape 4 by 1.

2) Apply the `softmax` function on all three tensors storing the results on `output_1`, `output_2` and `output_3`.

3) Print the results of `output_1`, `output_2` and `output_3`.

`@hint`


`@pre_exercise_code`
```{python}

```

`@sample_code`
```{python}
import torch
import torch.nn as nn
s = nn.Softmax()

input_1 = torch.randn(4,4)
output_1 = ___(input_1)
print(___, "\n")

input_2 = ___
output_2 = ___
print(___, "\n")

input_3 = ___
output_3 = ___
print(___)
```

`@solution`
```{python}
import torch
import torch.nn as nn
s = nn.Softmax()

input_1 = torch.randn(4,4)
output_1 = s(input_1)
print(output_1, "\n")

input_2 = torch.randn(1, 4)
output_2 = s(input_2)
print(output_2, "\n")

input_3 = torch.randn(4, 1)
output_3 = s(input_2)
print(output_3)
```

`@sct`
```{python}

```

---

## Finetuning a CNN

```yaml
type: NormalExercise
key: a005898fa6
xp: 100
```

In the last lecture we built a function takes in:
- A neural network
- An optimizer, called optimizer
- A loss function, called criterion
and trains the net on a given dataset.

Now, we are going to use that net to train a letter classifier. However, there might be a problem: our new training dataset is small, in addition to data being images of characters instead of letters.

Using the fine-tuning technique, we are going to show that training neural nets is possible even where we have a small dataset.

`@instructions`
- Instantiate and load the net from the serialized file `my_net.pth`.
- Replace the last layer with a randomly initialized linear layer which connects 7 * 7 * 512 features with 27 output classes.
- Put the net on train mode and then train it using `train_net()` function you built in the penultimate exercise.
- Put the net on eval mode and then test it using `test_net()` function you built in the last exercise.

`@hint`
- Net can be put on train mode using `.train()` function, and on eval mode using `.eval`.
- A linear layer can be instantiated using `nn.Linear(in_features, out_features)`.
- `load_state_dict` needs as argument a torch data-structure.
- Optimizer is called `optimizer`, while loss function is called `criterion`.

`@pre_exercise_code`
```{python}
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

np.random.seed(314)
indices = np.arange(10000)
np.random.shuffle(indices)
indices_train = indices[:50]
indices_test = indices[:20]

test_loader_emnist = torch.utils.data.DataLoader(
    datasets.MNIST('mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                   ])),
    batch_size=1, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices_train), num_workers=0)


test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('mnist', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                   ])),
    batch_size=1, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices_test), num_workers=0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # instantiate all 3 linear layers
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        self.fc = nn.Linear(7 * 7 * 512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 7 * 7 * 512)
        return self.fc(x)

model = Net()

# instantiate the Adam optimizer and Cross-Entropy loss function
optimizer = optim.Adam(model.parameters(), lr=3e-4)
` = nn.CrossEntropyLoss()

def train_net(model, optimizer, criterion):
    # mocked, it takes too long to train the net without a GPU
    if type(model) != type(Net()):
        raise TypeError('The first argument should be the model.')  
        

def test_net(model):
    # mocked, it takes too long to test the net
    return 0.57
```

`@sample_code`
```{python}
# Instantiate and load the net fom "my_net.pth"
model = ____
model.load_state_dict(____)

# Replace the last layer with a linear layer of size (7 * 7 * 512, 27)
model.fc = ____

# Put the net on train mode and train it using train_net()
model.____
train_net(____, ____, ____)

# Put the net on eval mode, test it and print the results
model.____
print("Accuracy of the net is: " + str(test_net(____)))
```

`@solution`
```{python}
# Instantiate and load the net fom "my_net_big.pth"
model = Net()
# model.load_state_dict(torch.load('my_net.pth'))

# Replace the last layer with a linear layer of size (7 * 7 * 512, 27)
model.fc = nn.Linear(7 * 7 * 512, 27)

# Put the net on train mode and train it using train_net()
model.train()
train_net(model, optimizer, criterion)

# Put the net on eval mode, test it and print the results
model.eval()
print("Accuracy of the net is: " + str(test_net(model)))
```

`@sct`
```{python}
success_msg("Well done! You just finished this PyTorch course.")
```
