# ============================================================================
# Enzyme 前向模式自定义规则 (Forward Mode / JVP)
# ============================================================================
"""
自定义前向模式规则：实现隐式微分

数学原理:
给定固定点条件: state* = f(state*, param, args...)
定义残差: F(state, param) = state - f(state, param, args...) = 0

隐式微分:
∂F/∂param + ∂F/∂state · ∂state*/∂param = 0
(I - J_state) · ∂state*/∂param = J_param

其中:
- J_state = ∂f/∂state|(state*, param)
- J_param = ∂f/∂param|(state*, param)

前向模式 (JVP):
(I - J_state) · dstate* = J_param · dparam
"""
function EnzymeRules.forward(
  config::FwdConfig,
  func::Const{typeof(fixed_point)},
  ::Type{<:Duplicated},
  f::Const,
  state::Duplicated,
  param::Duplicated,
  args::Const...; kw...)
  println("使用自定义前向模式规则 (隐式微分)")

  # 1. 前向传播：计算固定点 state*（使用默认参数）
  state_star = fixed_point(f.val, state.val, param.val, map(a -> a.val, args)...; kw...)

  # 确保 state_star 是数组（支持标量）
  state_is_scalar = !(state_star isa AbstractArray)

  state_star_vec = state_star
  state_is_scalar && (state_star_vec = [state_star])
  
  n = length(state_star_vec)

  # 2. 计算雅可比矩阵 J_state = ∂f/∂state|(state*, param)
  J_state = zeros(n, n)

  ## 先计算1个变量的

  for i in 1:n
    dstate = zeros(n)
    dstate[i] = 1.0

    state_dup = state_is_scalar ? Duplicated(state_star, dstate[1]) : Duplicated(copy(state_star_vec), dstate)
    param_const = Const(param.val)
    args_const = map(Const, map(a -> a.val, args))

    # 调用 f 的前向模式 AD
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
      result = autodiff(
        ForwardWithPrimal,
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

  # 4. 计算右端项 J_param · dparam
  state_const = state_is_scalar ? Const(state_star) : Const(state_star_vec)
  args_const = map(Const, map(a -> a.val, args))

  if state_is_scalar
    result = autodiff(
      ForwardWithPrimal,
      f.val,
      Duplicated,
      state_const,
      param,
      args_const...
    )
    # ForwardWithPrimal 返回 (derivative, primal)
    F_param_dot = [result[1]]

  else
    result_val = similar(state_star_vec)
    result_dval = zeros(n)
    autodiff(
      Forward,
      (out, s, p, a...) -> (out .= f.val(s, p, a...); nothing),
      Const,
      Duplicated(result_val, result_dval),
      state_const,
      param,
      args_const...
    )
    F_param_dot = result_dval
  end

  # 5. 求解线性系统 (I - J_state) · dstate* = F_param_dot
  A = I - J_state
  state_star_dot = A \ F_param_dot

  # 6. 返回结果（保持原始类型）
  if state_is_scalar
    return Duplicated(state_star, state_star_dot[1])
  else
    return Duplicated(state_star, state_star_dot)
  end
end
