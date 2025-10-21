# ============================================================================
# Enzyme 反向模式自定义规则 (Reverse Mode / VJP)
# ============================================================================
"""
Augmented primal: 前向传播 + 保存反向传播需要的数据

保存到 tape:
- state*: 收敛的固定点
- J_state: ∂f/∂state|(state*, param)
- param_val: param 的值
- args_vals: args 的值
- 其他元信息
"""
function EnzymeRules.augmented_primal(
  config::RevConfigWidth{1},
  func::Const{typeof(fixed_point)},
  ::Type{RT},
  f::Const,
  state::Union{Active, Duplicated},
  param::Duplicated,
  args::Const...
) where {RT<:Union{Active, Duplicated}}
  println("使用自定义反向模式规则 (augmented primal)")

  # 1. 前向传播：计算固定点（使用默认参数）
  # 处理不同类型的 state
  state_val = state isa Active ? state.val : state.val
  state_star = fixed_point(f.val, state_val, param.val, map(a -> a.val, args)...)

  # 处理标量情况
  state_is_scalar = !(state_star isa AbstractArray)
  if state_is_scalar
    state_star_vec = [state_star]
  else
    state_star_vec = state_star
  end
  n = length(state_star_vec)

  # 2. 计算并保存 J_state = ∂f/∂state|(state*, param)
  J_state = zeros(n, n)

  for i in 1:n
    dstate = zeros(n)
    dstate[i] = 1.0

    state_dup = state_is_scalar ? Duplicated(state_star, dstate[1]) : Duplicated(copy(state_star_vec), dstate)
    param_const = Const(param.val)
    args_const = map(Const, map(a -> a.val, args))

    if state_is_scalar
      result = autodiff(
        ForwardWithPrimal,
        f.val,
        Duplicated,
        state_dup,
        param_const,
        args_const...
      )
      # ForwardWithPrimal 返回 (derivative, primal)
      J_state[:, i] = [result[1]]
    else
      result_val = similar(state_star_vec)
      result_dval = zeros(n)
      autodiff(
        Forward,
        (out, s, p, a...) -> (out .= f.val(s, p, a...); nothing),
        Const,
        Duplicated(result_val, result_dval),
        state_dup,
        param_const,
        args_const...
      )
      J_state[:, i] = result_dval
    end
  end

  # 3. 保存数据到 tape
  tape = (
    state_star=state_star_vec,
    J_state=J_state,
    param_val=param.val,
    args_vals=map(a -> a.val, args),
    f_func=f.val,
    state_is_scalar=state_is_scalar
  )

  # 4. 返回
  primal_return = state_is_scalar ? state_star : copy(state_star_vec)

  # 对于 Active 返回类型，shadow 应该是 nothing
  if RT <: Active
    shadow_return = nothing
  else  # Duplicated
    shadow_return = state_is_scalar ? zero(state_star) : zeros(n)
  end

  return Enzyme.EnzymeRules.AugmentedReturn(
    primal_return,
    shadow_return,
    tape
  )
end

"""
Reverse pass: 反向传播梯度

数学原理:
给定上游梯度 dL/dstate*，计算 dL/dparam

伴随方程:
(I - J_state)^T · λ = dL/dstate*

然后:
dL/dparam = λ^T · J_param = λ^T · ∂f/∂param|(state*, param)
"""
function EnzymeRules.reverse(
  config::RevConfigWidth{1},
  func::Const{typeof(fixed_point)},
  dret,
  tape,
  f::Const,
  state::Union{Active, Duplicated},
  param::Duplicated,
  args::Const...; 
  kw...
)

  println("使用自定义反向模式规则 (reverse pass)")

  # 1. 从 tape 恢复数据
  state_star = tape.state_star
  J_state = tape.J_state
  param_val = tape.param_val
  args_vals = tape.args_vals
  f_func = tape.f_func
  state_is_scalar = tape.state_is_scalar

  n = length(state_star)

  # 2. 获取上游梯度 dL/dstate*
  if dret isa Active  # Active 情况
    dL_dstate_star = state_is_scalar ? [dret.val] : [dret.val]
  elseif dret isa Number  # 直接是数值
    dL_dstate_star = state_is_scalar ? [dret] : [dret]
  else  # Duplicated 情况
    dL_dstate_star = state_is_scalar ? [dret.dval] : dret.dval
  end

  # 3. 求解伴随方程: (I - J_state)^T · λ = dL/dstate*
  A_T = (I - J_state)'
  lambda = A_T \ dL_dstate_star

  # 4. 计算 dL/dparam = λ^T · J_param
  # 使用反向模式计算 J_param^T · λ

  # 构造一个辅助函数来计算 J_param 的列
  param_is_scalar = !(param_val isa AbstractArray)
  m = param_is_scalar ? 1 : length(param_val)

  param_grad = zeros(m)

  for i in 1:m
    dparam = param_is_scalar ? 1.0 : (p = zeros(m); p[i] = 1.0; p)

    param_dup = param_is_scalar ? Duplicated(param_val, dparam) : Duplicated(copy(param_val), dparam)
    state_const = state_is_scalar ? Const(state_star[1]) : Const(state_star)
    args_const = map(Const, args_vals)

    if state_is_scalar
      result = autodiff(
        ForwardWithPrimal,
        f_func,
        Duplicated,
        state_const,
        param_dup,
        args_const...
      )
      # ForwardWithPrimal 返回 (derivative, primal)
      J_param_col = [result[1]]
    else
      result_val = similar(state_star)
      result_dval = zeros(n)
      autodiff(
        Forward,
        (out, s, p, a...) -> (out .= f_func(s, p, a...); nothing),
        Const,
        Duplicated(result_val, result_dval),
        state_const,
        param_dup,
        args_const...
      )
      J_param_col = result_dval
    end

    # λ^T · J_param[:, i]
    param_grad[i] = dot(lambda, J_param_col)
  end

  # 5. 累加梯度到 param.dval
  if param_is_scalar
    param.dval += param_grad[1]
  else
    param.dval .+= param_grad
  end

  # 6. 返回
  # 对于 state: 如果是 Active，需要返回其梯度；如果是 Duplicated，返回 nothing（已修改 dval）
  # 对于固定点，state 的梯度为 0（因为我们在固定点处，不依赖初始 state）
  state_grad = if state isa Active
    zero(eltype(state_star))  # 返回标量 0
  else
    nothing  # Duplicated 类型，梯度已经在 dval 中
  end

  # 对于 param: 已经通过 param.dval 累加了梯度，所以返回 nothing
  return (nothing, state_grad, nothing, map(_ -> nothing, args)...)
end

