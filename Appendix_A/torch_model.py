import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([1.0]) # true label
x1 = torch.tensor([1.1]) # input feature
w1 = torch.tensor([2.2], requires_grad=True) # weight parameter
b = torch.tensor([0.0], requires_grad=True) # bias parameter

z = x1 * w1 + b # net input
a = torch.sigmoid(z) # activation function

loss = F.binary_cross_entropy(a, y) # loss function


grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_w1)
print(grad_L_b)

loss.backward()
print("--------------------------------")
print("w1.grad:")
print(w1.grad)
print("b.grad:")
print(b.grad)
print("--------------------------------")
