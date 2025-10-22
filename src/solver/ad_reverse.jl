"""
    fixed_point!(state_curr, f!, state, param, args...; kw...)

要求`f!`必须返回的是nothing，因为涉及逆向梯度。
"""
function fixed_point!(
  state_curr, f!, state, param, args...;
  tol::Float64=1e-6, nmax::Int=1000, norm_type::Real=2, verbose::Bool=false, kw...)

  state_prev = copy(state)
  f!(state_curr, state, param, args...; kw...) # 第一次迭代：state_curr = f(state)

  for iter in 1:nmax
    residual = norm(state_curr - state_prev, norm_type)

    if residual < tol
      verbose && println("Converged in $iter iterations")
      return nothing
    end

    state_prev .= state_curr # update
    f!(state_curr, state_prev, param, args...; kw...)
  end

  ϵ = norm(state_curr - state_prev, norm_type)
  @warn "Fixed point did not converge in $nmax iterations. Final residual: $ϵ"
  return nothing
end



function EnzymeRules.augmented_primal(
  config::RevConfig,
  func::Const{typeof(fixed_point!)},
  ::Type{Const{Nothing}},
  state_curr::Duplicated,
  f!::Const,
  state::Annotation,
  param::Duplicated,
  args...;
  kw...
)
  printstyled("[fixed_point!] custom `augmented_primal` \n", color=:blue, bold=true)
  
  func.val(state_curr.val, f!.val, state.val, param.val, args...; kw...) # 执行前向计算

  # 保存固定点和必要信息
  state_star = copy(state_curr.val) # 记录更新之后的信息
  tape = (state_star, f!.val, param.val, args, kw)

  return AugmentedReturn(nothing, nothing, tape) # primal, shadow, tape
end


# reverse：使用隐式函数定理
function EnzymeRules.reverse(
  config::RevConfig,
  func::Const{typeof(fixed_point!)},
  ::Type{Const{Nothing}},
  tape,
  output::Duplicated, # 存储的结果
  f!::Const,
  state::Annotation,
  param::Duplicated,
  args...; kw_unused...
)
  printstyled("[fixed_point!] custom `reverse` \n", color=:blue, bold=true)
  # println("In custom reverse for fixed_point!")

  state_star, f_saved!, param_saved, saved_args, kw = tape

  # 获取输出的 adjoint
  v = output.dval[:] # ∂F/∂a v

  # 隐式函数定理：
  # 解线性系统: (I - J_state)^T * λ = v
  # 其中 J_state = ∂f/∂state 在固定点处

  # 计算 Jacobian: J = ∂f/∂state
  n = length(state_star)
  J_state = zeros(n, n) # ∂F/∂x

  for i in 1:n
    dstate = make_zero(state_star)
    dstate[i] = 1.0
    d_st = make_zero(state_star)

    # 是否还需要自定义一个forward f_saved!，因为它不会自动填充
    autodiff(Forward, f_saved!,
      Const, # !important, return nothing 
      Duplicated(make_zero(state_star), d_st),  # 输出, state_next
      Duplicated(copy(state_star), dstate),               # 输入, state
      Const(param_saved),
      map(Const, saved_args)...
    )
    J_state[:, i] = d_st[:]
  end

  # 解线性系统: (I - J_state^T) * λ = v
  A = I - J_state' # 1 - ∂F/∂x

  λ = A \ v
  λ = reshape(λ, size(state_star))

  # 计算 ∂L/∂param = λ^T * ∂f/∂param
  # 使用反向模式计算
  d_param = make_zero(param_saved)

  # 这里需要执行一次，逆向求梯度，因素f必须返回nothing
  autodiff(Reverse, f_saved!,
    Const,
    Duplicated(similar(state_star), λ), # ∂f/
    Const(state_star),
    Duplicated(param_saved, d_param),
    map(Const, saved_args)...; kw...
  ) # ∂L/∂a += λ ∂F/∂a, 

  param.dval .+= d_param
  # make_zero!(state_curr.dval) # why? 
  return (nothing, nothing, nothing, nothing)
end
