# Graphs and Graph Neural Networks

## Graph Definition

A graph, denoted as `G`, can be formally defined as a set of vertices `V` and edges `E`, accompanied by an adjacency matrix `A`.

```plaintext
G = (V, E, A)
```

The adjacency matrix, `A`, is a vital representation of the graph structure which captures the pairwise relationships between vertices, forming a foundation for various graph-based analyses and computations.

```plaintext
- Aij = 1 if there is an edge between vertex i and vertex j.
- Aij = 0 otherwise.
```

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