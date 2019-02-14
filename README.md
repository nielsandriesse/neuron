### Overview

In this repo, I explore the use of an evolutionary algorithm for the optimization of a neural network’s hyperparameters, as well as the idea of having a group of neural networks work together to achieve better results than the group’s members could achieve individually.

Specifically, I evolve several sets of hyperparameters that work well, which I then use to train several neural networks to recognize digits in [the MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/). The idea is that each neural network will learn to recognize digits differently, improving performance when they’re later linked together as a group.

The neural network implementation itself is fairly basic. It’s a feedforward neural network that uses mini-batch based stochastic gradient descent, where the gradients are computed using backpropagation. I use the sigmoid function as the activation function and L2 regularization (weight decay) to reduce overfitting.

The hyperparameters being optimized are the layout of the neural network’s hidden layers (the number of hidden layers as well as the number of neurons in each hidden layer), the mini batch size, the learning rate and the regularization rate.

### Results & Discussion

The individual neural networks usually reach accuracies in the 96.5% - 97.5% range. That’s fairly good, though it could obviously be improved quite a bit further by using a more sophisticated neural network implementation. However, the point here is that it appears to be close to the maximum that can be achieved with the given setup. In other words, optimizing the hyperparameters using an evolutionary algorithm appears to work.

The average accuracy achieved by the group is in the 98.0% - 98.5% range. Again, this could be jacked up a bit more by improving the neural network implementation, but the point is that the group appears to consistently outperform the individual networks.

An example of what the sets of hyperparameters output by the evolutionary algorithm might look like:  

```
 Layout 1: [784 47 10], Mini Batch Size: 7, Learning Rate: 0.587307863476, Regularization Rate: 0.477624949415
Layout 2: [784 89 10], Mini Batch Size: 18, Learning Rate: 0.930009388195, Regularization Rate: 0.477624949415
Layout 3: [784 53 10], Mini Batch Size: 18, Learning Rate: 0.53817820585, Regularization Rate: 0.718179044034
Layout 4: [784 89 10], Mini Batch Size: 16, Learning Rate: 0.930009388195, Regularization Rate: 0.718179044034
Layout 5: [784 53 10], Mini Batch Size: 18, Learning Rate: 0.53817820585, Regularization Rate: 1.00397304786
Layout 6: [784 39 10], Mini Batch Size: 7, Learning Rate: 0.31251737242, Regularization Rate: 0.477624949415
Layout 7: [784 89 10], Mini Batch Size: 21, Learning Rate: 0.930009388195, Regularization Rate: 0.477624949415
Layout 8: [784 47 10], Mini Batch Size: 21, Learning Rate: 0.587307863476, Regularization Rate: 0.477624949415 ```

What’s important here is (obviously) that the hyperparameter sets work well, but also that they’re distinct. If they’re too similar, the idea of the networks learning different representations of the world is lost.

The evolutionary algorithm appears to prefer networks with a single hidden layer. This might be because it’s really optimal, but it could also very well be the case that deeper neural networks are getting filtered out because they take longer to train and therefore don’t perform well with the relatively low epoch count used in the evolutionary algorithm.

### Disclaimer

Clearly, this isn’t meant as any sort of formal investigation into hyperparameter optimization or neural network grouping. I did this for fun and to improve my own intuition about neural networks and their optimization. Feel free to play around with the code or use it for your own experiments, but take it with a grain of salt.
