# Graphs and Graph Neural Networks

## Graph Definition

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
- Minimization Target: $\sum A_{ij(y_i-y_j)^2}$, captures the 'out' cost.
- Equation: $y^T L y$, where $L$ is the Laplacian matrix and $y$ is a vector indicating node segments.
- onstraint Applied: $y^T y = 1$, ensures non-trivial solutions.
- Vector $y$: Represents the assignment of nodes to segments, dimension $n \times 1$.
- Solution Method: Eigen decomposition of $L$ identifies optimal partitioning.

