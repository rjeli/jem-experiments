# Joint Energy-based Models

Here is a rough implementation of [Your Classifier is Secretly an Energy Based Model and You Should Treat It Like One](https://arxiv.org/abs/1912.03263) (Grathwohl, Wang, Jacobsen, Duvenaud, Norouzi, Swersky, 2019).

The code is hacked together and there are certainly more than a few bugs. I expect to clean it up and fully replicate the paper's results in the coming week.

Currently (Wednesday 12/11) I have only trained a small resnet to 50% accuracy on 10% of CIFAR10. Although training becomes very unstable towards the end, the generated samples are promising.

## Plane
![](samples/plane0.png) ![](samples/plane1.png)
