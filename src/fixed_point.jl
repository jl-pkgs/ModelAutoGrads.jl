# https://enzyme.mit.edu/julia/dev/api/
using Enzyme
using Enzyme.EnzymeRules
using LinearAlgebra

# ============================================================================
# 通用固定点求解器
# ============================================================================

"""
    fixed_point(f, state, param, args...; kwargs...)

求解固定点方程: state* = f(state*, param, args...; kwargs...)

# 参数
- `f`: 迭代函数，签名为 `f(state, param, args...; kwargs...) -> state`
- `state`: 初始状态（要求解固定点的变量）
- `param`: 参数（通常需要对其求导）
- `args...`: 额外的位置参数（传递给 f）
- `kwargs...`: 额外的关键字参数，包括：
  - `tol`: 收敛容差（默认 1e-6）
  - `nmax`: 最大迭代次数（默认 1000）
  - `norm_type`: 范数类型（默认 2）
  - `verbose`: 是否打印信息（默认 false）

# 返回
- `state_star`: 收敛的固定点

# 示例
```julia
# 简单的标量迭代: x* = a * tanh(x*) + b
f(x, params) = params[1] * tanh(x) + params[2]
x_star = fixed_point(f, 0.0, [0.8, 0.2])

# 带额外参数
f(x, params, scale) = params[1] * tanh(scale * x) + params[2]
x_star = fixed_point(f, 0.0, [0.8, 0.2], 2.0)

# 带关键字参数
f(x, params; offset=0.0) = params[1] * tanh(x) + params[2] + offset
x_star = fixed_point(f, 0.0, [0.8, 0.2]; offset=0.1, tol=1e-8)
```
"""
function fixed_point(f, state, param, args...;
  tol::Float64=1e-6, nmax::Int=1000, norm_type::Real=2,
  verbose::Bool=false, kw...
)
  # 分离固定点求解的配置参数和传递给 f 的关键字参数
  state_prev = copy(state)
  state_curr = f(state, param, args...; kw...)

  for iter in 1:nmax
    residual = norm(state_curr - state_prev, norm_type)

    # verbose && println("Iter $iter: residual = $residual")
    if residual < tol
      verbose && println("Converged in $iter iterations")
      return state_curr
    end

    state_prev = copy(state_curr)
    state_curr = f(state_curr, param, args...; kw...)
  end

  ϵ = norm(state_curr - state_prev, norm_type)
  @warn "Fixed point did not converge in $nmax iterations. Final residual: $ϵ"
  return state_curr
end


export fixed_point
