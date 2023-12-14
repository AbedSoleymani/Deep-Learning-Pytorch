# Graphs and Graph Neural Networks

## Graph Definition and Graphical Data

A graph, denoted as $G$, can be formally defined as a set of vertices $V$ and edges $E$, accompanied by an adjacency matrix `A`.

$$
G = (V, E, A)
$$

The adjacency matrix, $A$, is a vital representation of the graph structure which captures the pairwise relationships between vertices, forming a foundation for various graph-based analyses and computations.

- $Aij = 1$ if there is an edge between vertex $i$ and vertex $j$.
- $Aij = 0$ otherwise.

Graphs are structured data. Many real-world datasets come in the form of graphs:
- Social networks
- Protein-interaction networks
- The World Wide Web
- Images are graphs, where each pixel represents a node and is connected via an edge to adjacent pixels. Images are actually Euclidean graphs which the vertices represent points in the plane, and each edge is assigned the length equal to the Euclidean distance between its endpoints
- Text is a graph. Each token is a node and is connected via an edge to the node that preceding it.

There are three forms of tasks performed by graphs:
- Graph-level task: Predicting the property of an entire graph, e.g., predict whether a molecule will bind to a receptor or not.
- Node-level task: In social networks, predicting the identity or role of each node within a graph, e.g., does this person belong to this community or not (classification) or does this person buy this camera (classification).
- Edge-level task: Making predictions or extracting information specifically at the level of individual edges within a graph. For instance, link prediction which involves predicting whether a connection (edge) between two nodes (entities) should exist in the graph. In the context of a recommender system, this could mean predicting whether a user will be interested in or likely to interact with a particular item, such as a movie, book, or product.

## Laplacian of a Graph
The Laplacian matrix $L$ of a graph provides Insights Into the graph's structure, Including its connectivity and the presence of clusters

$$
L= D - A
$$

where $D$ is the degree Matrix, a diagonal matrix with the degree of each vertex along the diagonal and $A$ is the adjacency Matrix, as defined above.

The degree matrix $D$ is a square matrix where each diagonal element $d_i$ represents the degree of the corresponding vertex $i$.

$$
D = \begin{bmatrix}
d_1 & 0 & \cdots & 0 \\
0 & d_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & d_n
\end{bmatrix}_{n \times n}
$$

where $d_i = \sum_{j} A_{ij}$. The degree $d_i$ is the sum of the elements in the $i$-th row of $A$, representing the number of edges connected to vertex $i$.

Laplacian has many useful properties when we do inference on graphs. One example is Graph Cut Optimization Problem.
- Objective: Find a cut that divides the graph into segments with minimal interconnections.
- Minimization Target: $\sum_{i,j} A_{ij}(y_i-y_j)^2$, captures the 'out' cost. The intuition behind this cost is that, if $y_i$ and $y_j$ are connected, $A_{ij}=1$ and these two nodes will be treated similarly. Otherwise (i.e., $A_{ij}=0$), we don not care! In this way, similar points should be close to each other and dissimilar points on the graph manifold will be far apart.
- Equation: It can be shown that minimizing $\sum_{i,j} A_{ij}(y_i-y_j)^2$ is equivalent to minimizing $y^T L y$, where $L$ is the Laplacian matrix and $y$ is a vector indicating node segments.
- Constraint Applied: $y^T y = 1$, ensures non-trivial solutions.
- Vector $y$: Represents the assignment of nodes to segments, dimension $n \times 1$.
- Solution Method: Eigen decomposition of $L$ identifies optimal partitioning.

Eigen Decomposition of Laplacian is in this way:
- Decomposed as $L = U \Lambda U^T$
- $U$: Orthonormal eigenvectors. Eigenvectors are orthogonal and normalized: $U^T U = I$
- $\Lambda$: Diagonal matrix with eigenvalues. Each diagonal entry is an eigenvalue that pairs with an eigenvector in $U$

Normalized Laplacian is defined as:

$$
    \widetilde{L} = D^{-\frac{1}{2}}(D - A)D^{-\frac{1}{2}}
$$

which can be simplified to:

$$
\widetilde{L} = I - D^{-\frac{1}{2}}AD^{-1}
$$

## Laplacian Eigenvectors and Fourier Analysis
The relation between Laplacian eigenvectors and Fourier analysis is rooted in spectral graph theory. In this context, the Laplacian matrix of a graph plays a role analogous to the Fourier transform in signal processing.

Fourier analysis is the decomposition of a signal into sinusoidal components. A signal is transformed to represent it as a sum of its frequency components. Sine and cosine functions serve as the basis for this transformation. These basis functions are orthogonal, ensuring a unique frequency representation.

**Values of nodes** in a graph can be considered as **feature** vector in a signal space. In this way, Laplacian eigenvectors serve a similar purpose to sinusoidal components in Fourier analysis by providing a basis for representing graph signals in the frequency domain.

A frequency-Like interpretation is that low eigenvalues correspond to "low-frequency" eigenvectors. These eigenvectors change slowly over the graph. They represent large-scale, smooth structures in the graph.
In contrast, high eigenvalues correspond to "high-frequency" eigenvectors. These eigenvectors change rapidly between connected nodes. They capture fine details or irregularities in the graph.

Suppose $x \in \mathbb{R}^n$ is a feature vector of all nodes of a graph where $x$ is the value of the ith node. The graph Fourier transform to a signal $x$ can be defined as:

$$
f(x) = U^T x = \hat{x}
$$

The inverse graph Fourier transform would be:

$$
f^{-1}(\hat{x}) = U\hat{x} = UU^Tx = x
$$

According to the Convolution Theorem, the graph convolution of the input signal $x$ with a filter $g \in \mathbb{R}$ is defined as:

$$
    (x * g) = \mathcal{F}^{-1}\left(\mathcal{F}(x) \cdot \mathcal{F}(g)\right)
$$

where $\mathcal{F}^{-1}$ denotes the inverse Fourier transform, $\mathcal{F}(x)$ and $\mathcal{F}(g)$ are the Fourier transforms of $x$ and $g$ respectively.

In the graph Fourier domain, this can be expressed as:

$$
    \mathcal{F}(x * g) = U^* (\mathcal{F}(x) \odot \mathcal{F}(g)) = U^* \text{diag}(\mathcal{F}(g)) U^* x
$$

where $\odot$ denotes element-wise multiplication and $U^*$ represents the conjugate transpose of the graph Fourier basis matrix $U$.

Therefore, in the vertex domain, the graph convolution is given by:

$$
    (x * g) = U \left(\text{diag}(U^* \mathcal{F}(g)) \cdot U^* x\right)
$$

where $\text{diag}(U^* \mathcal{F}(g))$ is a diagonal matrix with the elements of $U^* \mathcal{F}(g)$.
<<<<<<< HEAD

Now, we express the mathematics of a graph convolutional layer in a Spectral CNN.
Let $x$ be the input signal on the graph, $g$ be the filter (weight) associated with the graph convolutional layer, and $U$ be the matrix of Laplacian eigenvectors.

The graph convolution operation using Laplacian eigenvectors $U$ can be written as:

$$
H' = \sigma(U \Theta U^T H)
$$

where $x = H^{(0)}$, $g_\theta = \Theta$, $\sigma$ is an activation function, and $\Theta$ is a diagonal matrix with learnable parameters.

The presented model was Vanilla Spectral GNN with a Limitation: eigen-decomposition requires $O(n^3)$ computational complexity and can't be applied on large graphs, e.g., social networks.
||||||| 3d36471
=======

Now, we express the mathematics of a graph convolutional layer in a Spectral CNN.
Let $x$ be the input signal on the graph, $g$ be the filter (weight) associated with the graph convolutional layer, and $U$ be the matrix of Laplacian eigenvectors.

The graph convolution operation using Laplacian eigenvectors $U$ can be written as:

$$
H' = \sigma(U \Theta U^T H)
$$

where $x = H^{(0)}$, $g_\theta = \Theta$, $\sigma$ is an activation function, and $\Theta$ is a diagonal matrix with learnable parameters.

The presented model was Vanilla Spectral GNN with a bad Limitation: eigen-decomposition of Laplacian requires $O(n^3)$ computational complexity and can't be applied on large graphs, e.g., social networks.
From now, our goal is to devise a way to get around this problem and make this convolution or reformulate this convolution in a way that it does not depend on the decomposition of Laplacian.

## Chebyshev Polynomials of the First Kind
One approach is to approximates the Graph convolutional filter parameterized by $\theta$ or $g_\theta$ by Chebyshev polynomials of the diagonal matrix of eigenvalues $\Lambda$.
Considering the fact that:

$$
\cos(0) = 1
$$

$$
\cos(2\theta) = 2\cos^2(\theta) - 1
$$

$$
\cos(3\theta) = 4\cos^3(\theta) - 3\cos(\theta)
$$

By substituting $\cos(\theta) = x$, we obtain:

$$
T_0(x) = 1
$$

$$
T_1(x) = x
$$

$$
T_2(x) = 2x^2 - 1
$$

$$
T_3(x) = 4x^3 - 3x
$$

where $T_n(x)$ denotes the $n^{th}$ polynomial.
These polynomials have a nice recursive property as follow

$$
T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
$$

and are orthogonal on the interval $[-1,\ 1]$.
Now, we can write:

$$
g_{\theta} = \sum_i \theta_iT_i(\tilde{\Lambda})
$$

where $\tilde{\Lambda}$ is a scaled version of the eigenvalues of the Laplacian matrix which normalizes the eigenvalues to fall within $[-1, 1]$ in this way:

$$
\tilde{\Lambda} = \frac{2\Lambda}{\lambda_{\max}}-I_n
$$

Now, we will have:

$$
x*g_{\theta} = Ug_{\theta}U^Tx = \sum_i \theta_iUT_i(\tilde{\Lambda})U^Tx.
$$

Since $\tilde{\Lambda}$ and accordingly $T_i(\tilde{\Lambda})$ are diagonal and also $U$ is orthonormal, we can write:

$$
x*g_{\theta} = \sum_i \theta_iUT_i(\tilde{\Lambda})U^Tx = \sum_i \theta_i T_i(U \tilde{\Lambda}U^T)x = \sum_i \theta_i T_i(\tilde{L})x.
$$

We can compute $\tilde{L}$ without the eigendecomposition of Laplacian $L$. The scaled Laplacian $L$ is:

$$
\tilde{L} = \frac{2L}{\lambda_{\max}}-I_n.
$$

This algorithm is called ChebNet Graph Convolution. But, there is a big problem here: calculating $\lambda_{\max}$ requires eigen value decomposition which leads us to the same computational complexity problem! How we can get around this?

Please note that the whole task is learning weights $\theta_i$. We can simply neglect devision by $\lambda_{\max}$ and learn a scaled version of $\theta_i$ instead!

Now, we perform the first-order approximation:

$$
x * g_{\theta} = \sum_{i=0}^{k} \theta_i T_i(\tilde{L})x \approx \theta_0 T_0(\tilde{L})x + \theta_1 T_1(\tilde{L})x = \theta_0 x + \theta_1 \tilde{L} x
$$

To restrain the number of parameters and avoid over-fitting in GCN, we further assume $\theta = \theta_0 = \theta_1$. Now, we would have:

$$
x * g_{\theta} = \theta (I_n + \tilde{L}) x = \theta (I_n + \frac{2L}{\lambda_{\max}} - I_n) x = \frac{2\theta}{\lambda_{\max}}Lx = \Theta L x
$$

This empirically causes numerical instability to GCN. To address this problem, GCN applies a normalization trick to replace

$$
\tilde{A} = A+I  
$$

where $A$ is for aggregation of neighbouring nodes and $I$ is for considering the value of the node it self.
We know that the degree matrix can be defined as $D = \sum_j \tilde{A}_{ij}$

$$
\bar{A} = {D}^{-\frac{1}{2}} \tilde{A}{D}^{\frac{1}{2}}
$$

As a result, the compositional layer can be defined as:

$$
H^\prime = X*g_{\theta} = \sigma ( \bar{A}H\Theta )
$$

In summary, this expression represents a GNN layer's operation, where node features in the next layer are updated by aggregating information from neighboring nodes in the previous layer by a sort of matrix multiplication. The aggregation is controlled quantitatively by the graph convolutional operation $g_\theta$, and qualitatively the graph structure is considered through the adjacency matrix $\bar{A}$. In contrast, traditional NNs assume that data points are independent and identically distributed, making them less suitable for tasks involving graph-structured data, i.e., $\bar{A}=I_n$.

However, summing the contents of the neighbouring nodes will increase the scale of the output feature. As a result, we can update our layer update to a mean-pooling version rather than sum-pooling in this way:

$$
H^\prime = X*g_{\theta} = \sigma (D^{-1} \bar{A}H\Theta) \Rightarrow h^\prime_i = \sigma \left( \sum_{j \in N_i} \frac{1}{N_i} \Theta h_j \right)
$$

The node-wise update rule can be written as:

$$
h^\prime_i = \sigma \left( \sum_{j \in N_i} \frac{1}{\sqrt{|N_i||N_j|}} \Theta h_j \right).
$$

Now, we will take a step further and introduce Graph Attention Network (GAT). GAT adopts attention mechanisms to learn the relative weights between two connected nodes:

$$
h^\prime_i = \sigma \left( \sum_{j \in N_i} \alpha_{ij} \Theta h_j \right).
$$

where the attention weight $\alpha_{ij}$ measures the influence of node $j$ to node $i$.









>>>>>>> 8f484213c17dce4e093b590df73a9cfaef008d24
