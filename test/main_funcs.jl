using ComponentArrays


# 定义迭代函数
function f(state, params)
  (; A, b) = params
  state_next = A * tanh.(state) .+ b
  # state[1] = R[1] ## 迭代过程中state被修改
  return state_next
end


# 定义迭代函数
function f!(state_next, state, params, args...; kw...)
  (A, b) = params
  state_next .= A * tanh.(state) .+ b
  # state[1] = R[1] ## 迭代过程中state被修改
  return nothing
end


A = [0.7 0.2; 0.1 0.6]
b = [0.1 0.2]
params = ComponentArray(; A, b)

state_init = ones(Float64, 2, 2)
state_star = deepcopy(state_init)
