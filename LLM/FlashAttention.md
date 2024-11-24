# Flash Attenstion
## Forward Pass(standard)
if n = 3, dim = 4, the forward pass prcedure will be listed as followed:
$$
    Q\in \mathbb{R}^{3\times4}  \\
    K\in \mathbb{R}^{3\times4}  \\
    V\in \mathbb{R}^{3\times4} 
$$

$$
\begin{align}
    S & = Q \times K^T  &\in    \mathbb{R}^{3\times3} \tag{1} \\
    P & = softmax(S)    &\in    \mathbb{R}^{3\times3} \tag{2} \\ 
    O & = PV            &\in    \mathbb{R}^{3\times4} \tag{3}
\end{align}
$$
using matrix to specified the procedure tag1.
$$
\begin{bmatrix}
    q_{11} & q_{12} & q_{13} & q_{14} \\  
    q_{21} & q_{22} & q_{23} & q_{24} \\  
    q_{31} & q_{32} & q_{33} & q_{34} \\  
\end{bmatrix}  
\begin{bmatrix}  
    k_{11} & k_{21} & k_{31} \\  
    k_{12} & k_{22} & k_{32} \\  
    k_{13} & k_{23} & k_{33} \\  
    k_{14} & k_{24} & k_{34} \\  
\end{bmatrix}  
=\\
\begin{bmatrix}  
    q_{11} * k_{11} + q_{12} * k_{21} + q_{13} * k_{13} + q_{14} * k_{14} & q_{11} * k_{21} + q_{12} * k_{22} + q_{13} * k_{23} + q_{14} * k_{24} & q_{11} * k_{31} + q_{12} * k_{32} + q_{13} * k_{33} + q_{14} * k_{34} \\  
    q_{21} * k_{11} + q_{22} * k_{21} + q_{23} * k_{13} + q_{24} * k_{14} & q_{21} * k_{21} + q_{22} * k_{22} + q_{23} * k_{23} + q_{24} * k_{24} & q_{21} * k_{31} + q_{22} * k_{32} + q_{23} * k_{33} + q_{24} * k_{34} \\  
    q_{31} * k_{11} + q_{32} * k_{21} + q_{33} * k_{13} + q_{34} * k_{14} & q_{31} * k_{21} + q_{32} * k_{22} + q_{33} * k_{23} + q_{34} * k_{24} & q_{31} * k_{31} + q_{32} * k_{32} + q_{33} * k_{33} + q_{34} * k_{34} \\  
\end{bmatrix} = S
$$
We have that $S_{ij} = q_ik_j^T$ where $q_i$ and $k_j$ are the i-th and j-th rows of $\mathbf{Q}$ and $\mathbf{K}$ respectively.

using matrix to specified the procedure tag2.

$$
softmax(
\begin{bmatrix}
    s_{11} & s_{12} & s_{13} \\ 
    s_{21} & s_{22} & s_{23} \\ 
    s_{31} & s_{32} & s_{33} \\ 
\end{bmatrix}
)=
\begin{bmatrix}
    \frac{s_{11}}{s_{11} + s_{12} + s_{13}} & \frac{s_{12}}{s_{11} + s_{12} + s_{13}} & \frac{s_{13}}{s_{11} + s_{12} + s_{13}}\\
    \frac{s_{21}}{s_{21} + s_{22} + s_{23}} & \frac{s_{22}}{s_{21} + s_{22} + s_{23}} & \frac{s_{23}}{s_{21} + s_{22} + s_{23}}\\
    \frac{s_{21}}{s_{21} + s_{22} + s_{23}} & \frac{s_{22}}{s_{21} + s_{22} + s_{23}} & \frac{s_{23}}{s_{21} + s_{22} + s_{23}}\\
\end{bmatrix} = P
$$

using matrix to specified the procedure tag3.
$$
\begin{bmatrix}
    p_{11} & p_{12} & p_{13} \\ 
    p_{21} & p_{22} & p_{23} \\ 
    p_{31} & p_{32} & p_{33} \\ 
\end{bmatrix}
\begin{bmatrix}
    v_{11} & v_{12} & v_{13} & v_{14}\\ 
    v_{21} & v_{22} & v_{23} & v_{24}\\ 
    v_{31} & v_{32} & v_{33} & v_{34}\\ 
\end{bmatrix}
= \\
\begin{bmatrix}
    p_{11} * v_{11} + p_{12} * v_{21} + p_{13} * v_{31} & p_{11} * v_{12} + p_{12} * v_{22} + p_{13} * v_{32} & p_{11} * v_{13} + p_{12} * v_{23} + p_{13} * v_{33} & p_{11} * v_{14} + p_{12} * v_{24} + p_{13} * v_{34} \\
    p_{21} * v_{11} + p_{22} * v_{21} + p_{23} * v_{31} & p_{21} * v_{12} + p_{22} * v_{22} + p_{23} * v_{32} & p_{21} * v_{13} + p_{22} * v_{23} + p_{23} * v_{33} & p_{21} * v_{14} + p_{22} * v_{24} + p_{23} * v_{34} \\
    p_{31} * v_{11} + p_{32} * v_{21} + p_{33} * v_{31} & p_{31} * v_{12} + p_{32} * v_{22} + p_{33} * v_{32} & p_{31} * v_{13} + p_{32} * v_{23} + p_{33} * v_{33} & p_{31} * v_{14} + p_{32} * v_{24} + p_{33} * v_{34} \\
\end{bmatrix} = O
$$

let $v_j$ be the j-th row of $\mathbf{V}$, $o_i$ be the i-th row of $\mathbf{O}$, then the i-th colums of the output is:
$$
    o_i = P_{i:}\mathbf{V} = \sum_{j}P_{ij}v_{j} = \sum_{}\frac{e^{q_ik_j^T}}{L_i}v_j
$$

## Memory-efficient backward pass (v1.0)
The gradient of $\mathbf{dV}$ is:
$$
    \mathbf{dV} = \mathbf{P^TdO}
$$
Thus:
$$
    dv_j = \sum_{i}P_{ij}do_{i} \\
$$
let j = 1, the j-th row of $\mathbf{V}$
$$
\begin{align}
dv_1 &= \begin{bmatrix}
    dv_{11} &  dv_{12} & dv_{13} & dv_{14} 
\end{bmatrix} \\&= 
\begin{bmatrix}
    p_{11} + p_{21} + p_{31} &  p_{11} + p_{21} + p_{31} & p_{11} + p_{21} + p_{31} & p_{11} + p_{21} + p_{31}
\end{bmatrix} \\&= 
\begin{bmatrix} 
    \sum_{i}P_{i1}do_{i} & \sum_{i}P_{i1}do_{i} & \sum_{i}P_{i1}do_{i} & \sum_{i}P_{i1}do_{i} 
\end{bmatrix} \\&=
\begin{bmatrix}
    \sum_{i} \frac{e^{q_i k_1^T}}{Li} do_{i} & \sum_{i}\frac{e^{q_i k_1^T}}{Li}do_{i} & \sum_{i}\frac{e^{q_i k_1^T}}{Li}do_{i} & \sum_{i}\frac{e^{q_i k_j^T}}{Li} do_{i}
\end{bmatrix}
\end{align}
$$

Since we already comupted $L_i$, $dv_j$ can be computed without extra memory by repeated summing.

The gradient of $\mathbf{dQ}$ and $\mathbf{dQ}$:
go through the gradients of $\mathbf{dP}$ and $\mathbf{dS}$
$$
    dP_{ij} = do_iv_j^T
$$
for example, i = 1, j = 1:
$$
    dp_{11} = 
do_i\begin{bmatrix}
    v_{11} & v_{12} & v_{13} & v_{14}
\end{bmatrix}^T
$$

Recall that $P_{i:} = softmax(S_{i:})$. Using the fact the Jacobian of $y = softmax(x)$ is $diag(y) - yy^{T}$, we have that
$$
    dS_{i:} = (diag(P_{i:}) - P_{i:}P_{i:}^T)dP_{i:} = P_{i:} \circ dP_{i:} - (P_{i:}^TdP_{i:})P_{i:}
$$


# Deriviation of SoftMax
$$
   s_i = \frac{e^x_i}{\sum_{k=1}^{N}e_x^k }  \tag{1.1}
$$
$$
\begin{equation}
\frac{\partial s_i}{\partial x_j} = \begin{cases}
    -s_i^2 + s_i, & i = j;\\
    -s_i * s_j, & i != j.
\end{cases}
\end{equation}
$$

