using Enzyme
import Enzyme.EnzymeRules: augmented_primal, reverse
using .EnzymeRules
using LinearAlgebra

# 辅助函数：计算 Jacobian ∂f/∂state
function compute_jacobian_state(f, state, param, args...)
  n = length(state)
  J = zeros(n, n)

  for i in 1:n
    dstate = make_zero(state)
    dstate[i] = 1.0

    state_dup = Duplicated(copy(state), dstate)
    result = Duplicated(similar(state), make_zero(state))

    # 前向模式计算 Jacobian 的第 i 列
    autodiff(Forward, f, result, state_dup, Const(param), map(Const, args)...)
    J[:, i] = result.dval[:]
  end

  return J
end

# augmented_primal
function augmented_primal(
  config::RevConfigWidth{1},
  func::Const{typeof(fixed_point!)},
  ::Type{Const{Nothing}},
  state_curr::Duplicated,
  f::Const,
  state::Duplicated,
  param::Duplicated,
  args...;
  kw...
)
  println("In custom augmented_primal for fixed_point!")

  # 执行前向计算
  func.val(state_curr.val, f.val, state.val, param.val, args...; kw...)

  # 保存固定点和必要信息
  state_star = copy(state_curr.val)

  tape = (state_star, f.val, param.val, args)

  return AugmentedReturn(nothing, nothing, tape)
end

# reverse：使用隐式函数定理
function reverse(
  config::RevConfigWidth{1},
  func::Const{typeof(fixed_point!)},
  ::Type{Const{Nothing}},
  tape,
  state_curr::Duplicated,
  f::Const,
  state::Duplicated,
  param::Duplicated,
  args...
)
  println("In custom reverse for fixed_point!")

  state_star, f_saved, param_saved, saved_args = tape

  # 获取输出的 adjoint
  v = state_curr.dval[:]

  # 隐式函数定理：
  # 固定点条件: state* = f(state*, param)
  # 解线性系统: (I - J_state)^T * λ = v
  # 其中 J_state = ∂f/∂state 在固定点处

  # 计算 Jacobian
  J_state = compute_jacobian_state(f_saved, state_star, param_saved, saved_args...)

  # 解线性系统: (I - J_state^T) * λ = v
  A = I - J_state'
  λ = A \ v
  λ = reshape(λ, size(state_star))


  # 计算 ∂L/∂param = λ^T * ∂f/∂param
  # 使用反向模式计算
  state_const = Const(state_star)
  param_dup = Duplicated(param_saved, make_zero(param_saved))

  result_dup = Duplicated(similar(state_star), λ)

  println("k1")
  autodiff(Reverse, f_saved, result_dup, state_const, param_dup, map(Const, saved_args)...)

  println("k2")

  # 累加梯度到 param
  param.dval .+= param_dup.dval

  # 清零输出梯度
  make_zero!(state_curr.dval)

  return (nothing, nothing, nothing, nothing)
end
