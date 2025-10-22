using Enzyme
using ComponentArrays
using LinearAlgebra


Enzyme.API.runtimeActivity!(true)

function f!(state_next, state, params)
  # A, b = params.A, params.b
  (; A, b) = params
  R = A * tanh.(state) .+ b # 为何不能点赋值
  copyto!(state_next, R)
  return nothing
end

# function f!(state_next, state, params)
#   (; A, b) = params
#   # 关键：避免使用广播 .= 和 .+，改用循环
#   temp = tanh.(state)
#   mul!(state_next, A, temp)
#   @inbounds for i in eachindex(state_next)
#     state_next[i] += b[i]
#   end
#   return nothing
# end


begin
  # 使用不同的初始状态来区分各元素的贡献
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1 0.7; 0.5 0.4]
  params = ComponentArray(; A, b)

  state_init = [1.0 2.0; 3.0 4.0]  # 不同的值
  state_star = zeros(Float64, 2, 2)

  m = length(state_init)
  n = length(params)
end

begin
  res = make_zero(state_init)
  d_res = make_zero(state_init)

  d_state = make_zero(state_init)
  d_state[1] = 1.0

  autodiff(Forward, f!,
    Const,
    Duplicated(res, d_res),
    Duplicated(state_init, d_state),
    Const(params)
  )

  println("res = ", res)
  println("d_res = ", d_res)
end
