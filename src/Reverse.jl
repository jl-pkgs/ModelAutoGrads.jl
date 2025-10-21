# ============================================================================
# Enzyme 反向模式自定义规则 (Reverse Mode / VJP)
# ============================================================================
"""
反向模式隐式微分 (VJP)

数学原理:
F(state, param) = state - f(state, param, args...) = 0
隐式函数定理的伴随方程:
(I - J_state)ᵀ · λ = dL/dstate*
梯度计算:
dL/dparam = λᵀ · J_param

反向模式分两步:
1. augmented_primal: 前向传播 + 保存 J_state 到 tape
2. reverse: 求解伴随方程 + 计算参数梯度
"""

"""
Augmented primal: 前向传播 + 保存反向传播需要的数据
"""
function EnzymeRules.augmented_primal(
  config::RevConfigWidth{1},
  func::Const{typeof(fixed_point)},
  ::Type{RT},
  f::Const,
  state::Const{<:AbstractArray},  # 上一时刻的状态变量
  param::Duplicated,
  args::Const...;
  kw...
) where {RT<:Union{Active, Duplicated}}

  printstyled("使用自定义反向模式规则 (augmented primal)\n", color=:blue, bold=true)

  # 1. 计算固定点 state*
  state_star = fixed_point(f.val, state.val, param.val,
                          map(a -> a.val, args)...; kw...)
  n = length(state_star)
  args_const = map(a -> Const(a.val), args)

  # 2. 计算并保存 J_state = ∂f/∂state|(state*, param)
  J_state = zeros(n, n)

  for i in 1:n
    dstate = make_zero(state_star)
    dstate[i] = 1.0

    result_val = make_zero(state_star)
    result_dval = make_zero(state_star)

    autodiff(
      ForwardWithPrimal,
      (out, s, p, a...) -> (out .= f.val(s, p, a...); nothing),
      Const,
      Duplicated(result_val, result_dval),
      Duplicated(copy(state_star), dstate),
      Const(param.val),
      args_const...
    )
    J_state[:, i] = result_dval[:]
  end

  # 3. 保存数据到 tape
  tape = (
    state_star = state_star,
    J_state = J_state,
    param_val = param.val,
    args_vals = map(a -> a.val, args),
    f_func = f.val
  )

  # 4. 返回
  # 对于 Active 返回类型，shadow 应该是 nothing
  shadow_return = RT <: Active ? nothing : make_zero(state_star)

  return Enzyme.EnzymeRules.AugmentedReturn(
    state_star,
    shadow_return,
    tape
  )
end

"""
Reverse pass: 反向传播梯度

数学步骤:
1. 求解伴随方程: (I - J_state)ᵀ · λ = dL/dstate*
2. 计算参数梯度: dL/dparam = λᵀ · J_param (使用 VJP)
"""
function EnzymeRules.reverse(
  config::RevConfigWidth{1},
  func::Const{typeof(fixed_point)},
  dret,
  tape,
  f::Const,
  state::Const{<:AbstractArray},
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

  n = length(state_star)

  # 2. 获取上游梯度 dL/dstate*
  if dret isa Active
    dL_dstate_star = [dret.val]
  elseif dret isa Number
    dL_dstate_star = [dret]
  else  # Duplicated
    dL_dstate_star = dret.dval[:]
  end

  # 3. 求解伴随方程: (I - J_state)ᵀ · λ = dL/dstate*
  A_T = (I - J_state)'
  lambda = A_T \ dL_dstate_star

  # 4. 计算 dL/dparam = λᵀ · J_param (使用反向模式 VJP)
  # 准备输入
  args_const = map(Const, args_vals)

  # 准备输出的梯度（这是 lambda，即 dL/dstate*）
  output_grad = reshape(lambda, size(state_star))

  # 使用反向模式计算 VJP: λᵀ · J_param
  # 我们需要对 f(state_star, param, args...) 关于 param 求 VJP
  result_val = make_zero(state_star)
  result_grad = copy(output_grad)  # 输出的梯度

  param_grad_storage = make_zero(param_val)

  autodiff(
    Reverse,
    (out, s, p, a...) -> (out .= f_func(s, p, a...); nothing),
    Const,
    Duplicated(result_val, result_grad),
    Const(state_star),
    Duplicated(copy(param_val), param_grad_storage),
    args_const...
  )

  # 5. 累加梯度到 param.dval
  param.dval .+= param_grad_storage

  # 6. 返回
  # state 是 Const，所以返回 nothing
  # param 的梯度已经累加到 param.dval，所以也返回 nothing
  return (nothing, nothing, nothing, map(_ -> nothing, args)...)
end
