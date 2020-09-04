[![PyPI version shields.io](https://img.shields.io/pypi/v/disjunctive-nn.svg)](https://pypi.python.org/pypi/disjunctive-nn)
[![PyPI license](https://img.shields.io/pypi/l/disjunctive-nn.svg)](https://pypi.python.org/pypi/disjunctive-nn/)
[![PyPI license](https://travis-ci.com/tsitsimis/disjunctive-normal-networks.svg?branch=master)](https://travis-ci.com/tsitsimis/disjunctive-normal-networks.svg?branch=master)


# Disjunctive Normal Networks
A Disjunctive Normal Network (DNN) is a special type of Neural Network used for binary classification. It uses intersected convex polytopes (hyperdimensional polygons) to cover the feature space of positive samples. This allows DNNs to find rules in the form of constraining inequalities in feature space that resemble the rules present in Decision Trees (DTs).

 In 2D it can be seen as multiple convex polygons spread on the plane enclosing all positive samples while leaving negative samples outside. 

Based on paper:
> Mehdi Sajjadi, Mojtaba Seyedhosseini, Tolga Tasdizen (2014). Disjunctive Normal Networks. CoRR, abs/1412.8534.
 [\[pdf\]](https://arxiv.org/pdf/1412.8534.pdf)


## How to use disjunctive-nn
The disjuntive-nn package inherits from scikit-learn classes, and thus drops in neatly next to other sklearn transformers with an identical calling API.

```python
from disjunctive_nn import DisjunctiveNormalNetwork
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=1000, noise=0.2)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
    
dnn = DisjunctiveNormalNetwork(n_polytopes=2, m=4)
dnn.fit(X_train, y_train)

y_pred = dnn.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

## Installing
PyPI install:
```
pip install disjunctive_nn
```

Alternatively download the package, install requirements, and manually run the installer:
```
wget https://github.com/tsitsimis/disjunctive-normal-networks/archive/master.zip
unzip master.zip
rm master.zip
cd disjunctive-normal-networks-master

pip install -r requirements.txt

python setup.py install
```

## Benefits of Disjunctive Normal Networks
A Disjunctive Normal Network can be seen as an [Oblique Decision Tree](https://www.researchgate.net/publication/260972972_Oblique_Decision_Tree_Learning_Approaches_-_A_Critical_Review) (ODT) learned with backpropagation by minimizing an error function.

Oblique Decision Trees are a generalization of regular Decision Trees with rules that are multivariate resulting in dividing the feature space with boundaries non-parallel to the axes.

This allows DNNs to be much more interpretable (inequality rules on linear combination of features) than a vanilla Neural Network and to be trained with backpropagation.

Overall the main advantages of a DNN over a DT are:
- Uses polytopes instead of hyercubes
- Is trained with backpropagation and can be thus incorporated in any neural network topology as final or intermediate step
- Is less prone to overfitting (although this is a quick result shown by only some simple experiments)

## Theoretical background
### Disjunctive Normal Form
A Decision Tree segments the space in hypercubes by drawing axis-aligned hyperplanes. Each hyperbox encloses the points of one class to form the final decision function of the tree

<img src="./assets/decision-tree-boxes.jpg" height=200/>

In the case of binary classification, the interior of the hypercubes enclosing the positive samples can be seen as the subspace where a boolean function becomes True (1) and ouside is False (0).

<img src="./assets/decision-tree.png" height=200/>

For the tree of the above picture the associated boolean function (1 for positive class <img src="https://render.githubusercontent.com/render/math?math=c_%2B">, 0 for class <img src="https://render.githubusercontent.com/render/math?math=c_-">) is

<img src="https://render.githubusercontent.com/render/math?math=Y = ((x_1 < a_1) \cap (x_2 < a_2)) \cup ((x_1 < a_1) \cap (x_2 < a_2)^\prime \cap (x_3 < a_4)^\prime) \cup ((x_1 < a_1)^\prime \cap (x_4 < a_4)^\prime \cap (x_5 < a_5))">

This boolean function is written in [Disjunctive Normal Form](https://en.wikipedia.org/wiki/Disjunctive_normal_form) meaning that it is a union of intersections or an "OR of ANDs" (in terms of logic gates).

Here is when Disjunctive Normal Networks come into play to represent such boolean functions.

### Half-Spaces and Polytopes
A polytope is the generalization of a polygon and polyhedron in higher dimensions. It can be seen as the intersection of M half-spaces, where a half-space <img src="https://render.githubusercontent.com/render/math?math=H_i"> is defined as the sub-space where it holds <img src="https://render.githubusercontent.com/render/math?math=h_i(x) > 0">

<img src="./assets/polytope.png" height=140/>

Many such polytopes can be used as covers and optimized to enclose all positive samples in a binary classification problem:

<img src="./assets/polytopes.png" height=200/>

### Decision Function
A half-space can be expressed as a sigmoid function of a linear combination of the feature space
<img src="https://render.githubusercontent.com/render/math?math=h(x) = \sigma(w^Tx %2B a)">. The intersection of M half-spaces is their product (boolean AND) and forms a polytope <img src="https://render.githubusercontent.com/render/math?math=P_i">

<img src="https://render.githubusercontent.com/render/math?math=P_i = \displaystyle \product_{j=1}^{M} h_{ij}(x)">

Finally, the union of N polytopes forms the decision function <img src="https://render.githubusercontent.com/render/math?math=f(x)">. To calculate the union we could just add all the <img src="https://render.githubusercontent.com/render/math?math=P_i(x)"> together but in overlapping areas the result would be greater than 1. To tackle this, using the [DeMorgan](https://en.wikipedia.org/wiki/De_Morgan%27s_laws) rule <img src="https://render.githubusercontent.com/render/math?math=A \cup B = (A^\prime \cap B^\prime)^\prime"> the sum can be transformed to the product

<img src="https://render.githubusercontent.com/render/math?math=f(x) = 1 - \displaystyle \product_{i=1}^{N}(\product_{j=1}^{M} 1 - h_{ij}(x))">

In the above expression we replace boolean negation of a variable <img src="https://render.githubusercontent.com/render/math?math=A"> with <img src="https://render.githubusercontent.com/render/math?math=1-A"> and the boolean AND with a product. 

The function <img src="https://render.githubusercontent.com/render/math?math=f(x)"> is then optimized with gradient descent.


## Examples and benchmarking
DNNs were tested on 2D synthetic datasets and compared to Decision Trees which is the closest classifier in terms of complexity and shape of decision function. The point of this experiment is to illustrate the nature of decision boundaries of the 2 classifiers. This should be taken with a grain of salt, as the performance does not necessarily carry over to real datasets.

The accuracy corresponds to the test set after splitting the dataset in train and test set. DNN parameters (N: number of polytopes, M: number of half-spaces per polytope) are set through experimentation.

<table>
    <th>Dataset</th>
    <th>DNN</th>
    <th>DT</th>
    <th>DNN Parameters</th>
    <tr>
        <td>Moons</td>
        <td><b>0.98</b></td>
        <td>0.96</td>
        <td>N=2, M=4</td>
    </tr>
    <tr>
        <td>Circles</td>
        <td><b>0.98</b></td>
        <td>0.96</td>
        <td>N=1, M=4</td>
    </tr>
    <tr>
        <td>Spirals</td>
        <td><b>0.99</b></td>
        <td>0.96</td>
        <td>N=20, M=10</td>
    </tr>
</table>

The below plots show the 2 models' decision function when trained on the whole dataset. The purpose is to show how well the models memorize (overfit) the training set.

### Moons
<img src="./assets/moons-experiments.png" width="100%"/>

### Circles
<img src="./assets/circles-experiments.png" width="100%"/>

### Spirals
<img src="./assets/spirals-experiments.png" width="100%"/>


The overall observation is that DNNs provide much **smoother decision boundaries** and overfit less on training data.
