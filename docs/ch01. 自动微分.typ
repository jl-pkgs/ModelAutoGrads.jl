#import "@local/modern-cug-report:0.1.3": *
// #let boxed = markrect.with(outset: (x: 0.3em, y: 0.3em))

#let pi = markhl.with(color: yellow)
#let pj = markhl.with(color: red)

#show: doc => template(doc, footer: "CUG水文气象学2025", header: "")

#show math.equation: set text(size: 12.5pt)
#show math.equation.where(block: true): doc => {
  set text(size: 12.5pt)
  set par(spacing: 0.7em, leading: 0.8em)
  doc
}

#set math.mat(delim: "[")
#let hat(x) = $overline(#x)$

= 1 *Auto Gradient*

在编写基于物理过程的混合模型时，有时需要自定义梯度（`custom_jvp`或`custom_vjp`），需要对Forward和Reverse梯度有所了解。


#beamer-block[推导过程见：https://chatgpt.com/c/68f5a76f-b9dc-8320-9ae6-a85861b9482d]

有可微函数$f = f(x_1, x_2, ..., x_n): RR^{n} -> RR^{m}$，$J = pdv(f, x) in RR^{m times n}$。
$
  J = pdv(f, x) =
  mat(
    pdv(f_1, x_1), …, pdv(f_1, x_j), …, pdv(f_1, x_n);
    ⋮, , ⋮, , ⋮;
    pdv(f_i, x_1), …, pdv(f_i, x_j), …, pdv(f_i, x_n);
    ⋮, , ⋮, , ⋮;
    pdv(f_m, x_1), …, pdv(f_1, x_j), …, pdv(f_m, x_n)
  )
$


$ "Model:" y = f(x) $

$ "Loss:" L = l(y) $

$grad_y l^T = pdv(L, y)$，全微分$d L$：

$
  d L = grad_y l^T thin d y, quad d y = J thin d x
$

$
  d L & = grad_y l^T thin (J thin d x), quad   & "\"Forward\"", \
  d L & = (J^T thin grad_y l)^T thin d x, quad &  "\"Reverse\""
$

因此，$grad_x l = pdv(L, x) = J^T thin grad_y l$。

// #pagebreak()

== 1.1 Forward (JVP)

- $dot(x)$：方向导数（directional derivative），可理解为$d x$，表示输入变化引起的中间量变化

$
  d y = grad f(x) d x \
  boxed(dot(y) = grad f(x) dot dot(x) = J dot v)
$

“输出的变化率 = 雅可比矩阵 × 输入的变化率”，Jacobian-vector product (JVP)

#beamer-block[`jacfwd`前向(jvp)：$delta y = J thin delta(x)$，循环$n$次]

$
  pdv(f, x_j)
  = J dot e_j
  = mat(
    pdv(f_1, x_1), …, pj(pdv(f_1, x_j)), …, pdv(f_1, x_n);
    ⋮, , ⋮, , ⋮;
    pdv(f_i, x_1), …, pj(pdv(f_i, x_j)), …, pdv(f_i, x_n);
    ⋮, , ⋮, , ⋮;
    pdv(f_m, x_1), …, pj(pdv(f_1, x_j)), …, pdv(f_m, x_n)
  ) dot mat(0; ⋮; 1; ⋮; 0)
  = mat(pj(pdv(f_1, x_j)); ⋮; pj(pdv(f_i, x_j)); ⋮; pj(pdv(f_m, x_j))),quad e_j in RR^{n times 1}
$

// $ J = [pdv(f, x_1), pdv(f, x_2), ..., pdv(f, x_n)] $
#v(0.8em)

== 1.2 Reverse (VJP)

- $hat(x)$：灵敏度（sensitivity），表示输出变化对输入的影响

#h(2em)
let #h(0.4em) $boxed(bold(hat(x) = pdv(L, x)))$ #h(0.25em), $hat(y) = pdv(L, y)$，

$ pdv(L, x) = pdv(L, y) pdv(y, x) $

$ boxed(overline(x) = overline(y) dot grad f(x) = v dot J) $

“输入的灵敏度 = 输出灵敏度 × 局部导数”，Vector-Jacobian product (VJP)

#beamer-block[`jacrev`反向(vjp)：$delta x = delta(y)^T J$，循环$m$次]

输出方向上的线性组合，$pi(v_i in RR^{m times 1})$：

$
  pdv(f_i, x) = v_i^T J 
  = mat(0, ..., 1, ..., 0) dot mat(
    pdv(f_1, x_1), …, pdv(f_1, x_j), …, pdv(f_1, x_n);
    ⋮, , ⋮, , ⋮;
    pi(pdv(f_i, x_1)), …, pi(pdv(f_i, x_j)), …, pi(pdv(f_i, x_n));
    ⋮, , ⋮, , ⋮;
    pdv(f_m, x_1), …, pdv(f_1, x_j), …, pdv(f_m, x_n)
  )
  // = mat(pj(pdv(f_1, x_j)); ⋮; pj(pdv(f_i, x_j)); ⋮; pj(pdv(f_m, x_j))),quad e_j in RR^{n times 1}

  = mat(pi(pdv(f_i, x_1)), …, pi(pdv(f_i, x_j)), …, pi(pdv(f_i, x_n)))
$

// #pagebreak()

= 2 *应用情景*

== 2.1 模型框架

```julia
# 模拟海温
function model(perturbations, state_init, param, args...; kw...)
  state_final = runsteps!(perturbations, state_init, param, args...; kw...)
  loss = loss_fun(state_obs, state_final)
  
  return state_final # scenario 1 or 3
  return loss        # scenario 2
end
```

#beamer-block()[*`U`*: 扰动变量perturbations; #h(1em) *`S`*: 状态变量state; #h(1em) *`L`*: 损失函数loss]

- S1. 数据同化设定*扰动状态*，求∂L/∂U，其他变量认为常数：
  
  `perturbations_next = perturbations_cur - η ∂L/∂U`

- S2. 模型调优设定*模型参数*，求∂L/∂P，其他变量认为常数：
  
  `param_next = param_cur - η ∂L/∂P`

- S3. *灵敏度分析*，求∂S/∂P，根据梯度分析敏感性，无目标函数。

理论细节：https://claude.ai/chat/f4b4bf78-2226-4e9d-9bdc-f806530da0ec

海温案例：https://enzyme.mit.edu/julia/stable/generated/box/

#v(0.6em)

== 2.2 参数调优

- 牛顿梯度降：#h(7em) $x_{n+1} = x_{n} - eta f(x) / (f^' (x))$

- 梯度降：#h(9em) $x_{n+1} = x_{n} - eta grad L(x)$ <eq_grad_optim>

#box-blue[
  L最小时，一阶导为0：$g(x) = grad L(x) = 0$；L的二阶导$g'(x) approx eta$，其中$eta$为学习速率参数。
  L最小时，一阶导为0：$g(x) = grad L(x) = 0$；L的二阶导$g'(x) approx eta$，其中$eta$为学习速率参数。
  https://claude.ai/chat/f4b4bf78-2226-4e9d-9bdc-f806530da0ec
]

*根据梯度降理论（式#[@eq_grad_optim]），更新模型扰动U或参数P。*

// $ x_{t+1} = f(x_{t}, w) $
// 其中$x_{t}$为状态变量，$w$为模型参数；
// - 敏感性分析
// - 根据`loss`优化param

= 3 *数学推导*

== 3.1 前向传播

有可微函数$f = f(x_1, x_2, ..., x_n)$，每个输入$x_i$对$f$的影响，即梯度$pdv(f, x_i)$。

假设$f$是由很多中间变量$w = (w_1, w_2, ..., w_{m})$计算得到，采用中间变量替换后$f(x_1, x_2, ..., x_n)==>F(w_1, ..., w_m)$。

$
  w_1 & = g_1(x_1, dots, x_n) \
  w_2 & = g_2(w_1; x_1, dots, x_n) \
  w_3 & = g_3(w_1, w_2; x_1, dots, x_n) \
      & dots.v \
  w_m & = g_m(w_1, ..., w_{m-1}; x_1, ..., x_n) \
    f & = F(w_1, ..., w_m;)
$

根据链式法则：

$
  pdv(w_1, x_i) & = pdv(g_1, x_i) \
  pdv(w_2, x_i) & = pdv(g_2, x_i) + pdv(g_2, w_1) pdv(w_1, x_i) \
  pdv(w_3, x_i) & = pdv(g_3, x_i) + pdv(g_3, w_2) pdv(w_2, x_i) + pdv(g_3, w_1) pdv(w_1, x_i)
$

$pdv(w_j, x_i)$的一般式为：

$ pdv(w_j, x_i) = pdv(g_j, x_i) + sum_(k=1)^(j-1) pdv(g_j, w_k) pdv(w_k, x_i) $

$pdv(f, x_i)$的一般式为：

$ pdv(f, x_i) = sum_{j=1}^{m} pdv(F, w_j) pdv(w_j, x_i) $ <eq_fwd>


== 3.2 反向传播

反向传播同样采用式#[@eq_fwd]，但事先给出每个中间变量的灵敏度(adjoint) $overline(w)_k = pdv(f, w_k)$，之后根据链式法则，解出每个$x_j$的梯度：

$ x_i = sum_k overline(w)_k pdv(w_k, x_i) $




= 4 *Jax*

== 4.1 *VegCan*

$ x = F(x; w) $

$ x_{m+1} = F(x_m; w) $

$ pdv(x, w) = dv(F(x, w), w) $

$
  pdv(x, w) = pdv(F, x) pdv(x, w) + pdv(F, w) ==>
  pdv(x, w) [1 - pdv(F, x)] = pdv(F, w)
$

$ pdv(x, w) = [I - pdv(F, x)]^{-1} pdv(F, W), I in RR^{n_x times n_x} $

引入JVP

$ [I - pdv(F, x)] pdv(x, w) v = pdv(F, w) v, quad pdv(x, w) v in RR^{n_x times 1} $

JVP $pdv(x, w) v$通过，下述线性方程求得：

$ A pdv(x, w) v = b, A = [I - pdv(F, x)], b = pdv(F, w) v $

#pagebreak()

== 4.2 fixed_point

$ x^* = f(a, x^*) $

#beamer-block[其中$a$是模型参数param，$x$是状态变量state。]

$ pdv(x^*, a) = pdv(f, a) + pdv(f, x) pdv(x^*, a) $

$ (I - pdv(f, x)) pdv(x^*, a) = pdv(f, a) $

$ pdv(x^*, a) = (I - pdv(f, x))^{-1} pdv(f, a) $

在反向传播中，已经计算得出$overline(x)^*$。

$
  overline(a) = pdv(L, a) = pdv(L, x) pdv(x, a) = overline(x)^* pdv(x, a)
  = overline(x)^* (I - pdv(f, x))^{-1} pdv(f, a)
$

*$pdv(f, x)$的解法：* 正向求解梯度，已知$d x$ 求$d y$ (JVP)

```julia
J_state = zeros(n, n) # ∂F/∂x
for i in 1:n
  dstate = make_zero(state_star)
  dstate[i] = 1.0

  _input = Duplicated(copy(state_star), dstate)
  _output = Duplicated(make_zero(state_star), make_zero(state_star))

  autodiff(Forward, f_saved!,
    _output, _input, Const(param.val), map(Const, saved_args)...; kw...)
  J_state[:, i] = _output.dval[:]
end
```

定义辅助变量，

$
  lambda = u^* = overline(x)^* (I - pdv(f, x))^{-1}, quad overline(a) = u^* pdv(f, a)
$
#box-blue[
  1. 这里不采用矩阵的方式进行求解，因为$pdv(f, x) in RR^{n times n}$计算代价高。
  2. 迭代法，每次只计算一个雅可比-向量积，计算、存储高效。
]

展开：#h(10em) $u^* = overline(x)^* + u^* pdv(f, x)$

// 反向模式下，处理的是行向量，
// $ u^* = overline(x)^* + (pdv(f, x))^T u^* $

通过迭代，可以求得$u^*$。

已知$u^*$的情况下，$boxed(overline(a) = u^* pdv(f, a))$，采用逆向求解，已知灵敏度，逆向求梯度。

```julia
# VJP, ∂F/∂a, v = λ
autodiff(Reverse, f_saved!,
    Const,
    Duplicated(similar(state_star), λ),
    Const(state_star),
    Duplicated(param_saved, d_param), 
    map(Const, saved_args)...; kw...
  ) # ∂L/∂a += λ ∂F/∂a, 
```

// #raw(read("./fix_point.py"), block:true, lang: "python")

// // *用途：*
// #mitex(
//   `$$
// L = \mathbf{v}^T \mathbf{f}(\mathbf{x}) = \sum_{i=1}^{m} v_i f_i(\mathbf{x})
// $$`,
// )
// $ dv(L, x) = v^T pdv(f, x) = v^T J $

#pagebreak()

=== 4.2.1 示例
```python
from jax import jacfwd as builtin_jacfwd

def our_jacfwd(f):
    def jacfun(x):
        _jvp = lambda s: jvp(f, (x,), (s,))[1]
        Jt = vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun

assert jnp.allclose(builtin_jacfwd(f)(W), our_jacfwd(f)(W)), 'Incorrect forward-mode Jacobian results!'
```


```python
from jax import jacrev as builtin_jacrev

def our_jacrev(f):
    def jacfun(x):
        y, vjp_fun = vjp(f, x)
        # Use vmap to do a matrix-Jacobian product.
        # Here, the matrix is the Euclidean basis, so we get all
        # entries in the Jacobian at once.
        J, = vmap(vjp_fun, in_axes=0)(jnp.eye(len(y)))
        return J
    return jacfun
assert jnp.allclose(builtin_jacrev(f)(W), our_jacrev(f)(W)), 'Incorrect reverse-mode Jacobian results!'
```

// #pagebreak()

== 4.3 `custom_jvp`

$
  "tangent"_"out" = f_"jvp" ( (x, y), (x_"dot", y_"dot")) =
  J_f(x, y) dot mat(delim: "[", x_"dot"; y_"dot")
$

```python
@jax.custom_jvp
def f(x, y):
  return jnp.sin(x) * y

@f.defjvp
def f_jvp(primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = f(x, y)
  tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
  return primal_out, tangent_out
```

== 4.4 `custom_vjp`

```python
@jax.custom_vjp
def f(x, y):
  return jnp.sin(x) * y

def f_fwd(x, y):
  return f(x, y), (jnp.cos(x), jnp.sin(x), y)

def f_bwd(res, g):
  cos_x, sin_x, y = res
  return (cos_x * g * y, sin_x * g)

f.defvjp(f_fwd, f_bwd)
```

// == 1.6 为何需要vjp
// 在正向计算非常耗时时，发挥作用。
// $
//   g = pdv(L, f) \
//   pdv(L, x) =  pdv(L, f) pdv(f, x) = g pdv(f, x) \
//   pdv(L, y) =  pdv(L, f) pdv(f, y) = g pdv(f, y) \
// $
// 简化梯度计算
// 正向：$ d f = pdv(f, x) d x + pdv(f, y) d y$
// // $
// //   "grad"(f, (0, 1))(x, y)
// // $

= *References* // <!-- omit in toc -->

1. https://enzymead.github.io/Enzyme.jl/dev/

2. https://adrianhill.de/julia-ml-course/L6_Automatic_Differentiation/
