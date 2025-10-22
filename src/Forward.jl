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


"""
前向模式隐式微分 (JVP)

数学原理:
F(state, param) = state - f(state, param, args...) = 0
隐式函数定理: (I - J_state) · dstate* = J_param · dparam

前向模式只需计算 JVP:
dstate* = (I - J_state)⁻¹ · (J_param · dparam)
"""
function EnzymeRules.forward(
  config::FwdConfig,
  func::Const{typeof(fixed_point)},
  ::Type{<:Duplicated},
  f::Const,
  state::Const{<:AbstractArray}, # 上一时刻的状态变量
  param::Duplicated,
  args::Const...; kw...)

  printstyled("使用自定义前向模式规则 (隐式微分)\n", color=:blue, bold=true)

  # 1. 计算固定点 state*
  state_star = fixed_point(f.val, state.val, param.val,
    map(a -> a.val, args)...; kw...)
  n = length(state_star)
  args_const = map(a -> Const(a.val), args)

  # 2. 计算 J_state = ∂f/∂state|(state*, param)
  J_state = zeros(n, n) ## 向前求导
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

  # 3. 计算 J_param · dparam (一次 JVP)
  result_val = make_zero(state_star)
  result_dval = make_zero(state_star)

  autodiff(
    Forward,
    (out, s, p, a...) -> (out .= f.val(s, p, a...); nothing),
    Const,
    Duplicated(result_val, result_dval),
    Const(state_star),
    param,  # 直接使用 dparam
    args_const...
  )
  b = result_dval[:] # ∂F/∂P · v

  # 4. 求解 (I - J_state) · dstate* = result_dval
  A = I - J_state
  dstate_star = A \ b # ∂X/∂P · v 
  dstate_star = reshape(dstate_star, size(state_star))

  return Duplicated(state_star, dstate_star)
end
