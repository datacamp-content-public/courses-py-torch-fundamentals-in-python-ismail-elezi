---
title: 'Chapter Title Here'
description: 'Chapter description goes here.'
---

## Insert exercise title here

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

Let's see the differences between neural networks which apply ReLU and those which do not apply ReLU. We have the code given here which randomly initializes the input called input_layer, and three sets of weights, called weight_1, weight_2 and weight_3.

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

On the previous exercise, it is possible that you got a vector of zeros in the output. The reason for this is that if a unit is connected with the previous layer only with negative weights, then that unit will become 0 after applying ReLU on it. Then that unit won't have any effect on the next layer, and so on. When a unit has value 0, that unit is considered dead. Based on some studies, in modern neural networks circa 30% of units are dead units.

In order to fix this, there is a simple trick. You can replace ReLU with leaky ReLU which implements the function `leaky_ReLU(x) = (a*x, x)` where a is a small positive number, meaning that negative values instead of being set to 0 are set to a small number instead (the number being a times original value).

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

On the previous exercises, we didn't insert non-linearity on the final layer. On neural networks, the final layer typically has a different version of non-linearity, whose purpose (in addition to adding non-linearity) is to transform the output values into a probability distribution. This allows the network to also point out the confidence of its prediction.

This type of function is called `softmax` function (`logistic` or `sigmoid` function on binary case) and is implemented on `torch.nn` module. Let us see how it works.

`@instructions`
1) Create three torch random tensors. The first one is called `input_1` and has shape 4 by 4, the second one is called `input_2` and has shape 1 by 4, the third one is called `input_3` and has shape 4 by 1.

2) Apply the softmax function on all three tensors storing the results on `output_1`, `output_2` and `output_3`.

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
print(output_1, "\n")

input_2 = ___
output_2 = ___
print(output_2, "\n")

input_3 = ___
output_3 = ___
print(output_3)
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
