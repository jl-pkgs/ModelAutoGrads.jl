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
  state::Const{<:AbstractArray},
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
  # primal: 如果 needs_primal(config)=true 则返回 state_star，否则 nothing
  # shadow: 如果 needs_shadow(config)=true 则返回 zero，否则 nothing
  primal_return = EnzymeRules.needs_primal(config) ? state_star : nothing
  shadow_return = EnzymeRules.needs_shadow(config) ? make_zero(state_star) : nothing

  return Enzyme.EnzymeRules.AugmentedReturn(
    primal_return,
    shadow_return,
    tape
  )
end

"""
Reverse pass: 反向传播梯度

数学步骤:
1. 求解伴随方程: (I - J_state)ᵀ · λ = dL/dstate*
2. 计算参数梯度: dL/dparam = λᵀ · J_param (使用 VJP)

关键修正:
- reverse 函数不应该直接修改 param.dval
- 应该返回梯度值，Enzyme 会自动累加
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
  # dret 可能是:
  # - Active: dret.val 是标量
  # - Duplicated: dret.dval 是数组
  # - 直接的数值
  dL_dstate_star = if dret isa Active
    fill(dret.val, n)  # 标量广播
  elseif dret isa Duplicated
    copy(dret.dval[:])
  elseif dret isa AbstractArray
    copy(dret[:])
  else
    fill(dret, n)
  end

  # 3. 求解伴随方程: (I - J_state)ᵀ · λ = dL/dstate*
  A_T = (I - J_state)'
  lambda = A_T \ dL_dstate_star

  # 4. 计算 dL/dparam = λᵀ · J_param (使用反向模式 VJP)
  args_const = map(Const, args_vals)

  # 准备输出的梯度（这是 lambda，即伴随变量）
  output_grad = reshape(lambda, size(state_star))

  # 使用反向模式计算 VJP: λᵀ · J_param
  result_val = make_zero(state_star)
  result_grad = copy(output_grad)

  # 关键: 用 zero 初始化，让 Enzyme 自动累加
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

  # 5. 返回梯度
  # 返回值顺序必须与参数顺序一致:
  # (f, state, param, args...)
  #
  # - f 是 Const，返回 nothing
  # - state 是 Const，返回 nothing  
  # - param 是 Duplicated，返回计算出的梯度
  # - args 都是 Const，返回 nothing
  
  return (
    nothing,              # f::Const 的梯度
    nothing,              # state::Const 的梯度
    param_grad_storage,   # param::Duplicated 的梯度
    map(_ -> nothing, args)...  # args::Const... 的梯度
  )
end
