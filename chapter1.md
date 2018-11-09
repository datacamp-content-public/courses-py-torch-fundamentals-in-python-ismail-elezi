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
1) Multiply the input_layer with weight_1 and store the result in hidden_1, then multiply hidden_1 with weight_2 and store the result in hidden_2
2) Multiply hidden_2 with weight_3 and print the result
3) Multiply weights_1 with weights_2 and store the result in weight_composed_1, then multiply weight_composed_1 with weight_3 and store the result in weight.
4) Multiply input_layer with weight and print the results.

`@hint`


`@pre_exercise_code`
```{python}
import torch
import torch.nn as nn

relu = nn.ReLU()
```

`@sample_code`
```{python}

```

`@solution`
```{python}
hidden_1 = torch.matmul(input_layer, weight_1)
hidden_2 = torch.matmul(hidden_1, weight_2)\

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


`@hint`


`@pre_exercise_code`
```{python}

```

`@sample_code`
```{python}

```

`@solution`
```{python}

```

`@sct`
```{python}

```
