using ModelAutoGrad
using Enzyme
using Enzyme: autodiff


# 定义迭代函数
function f(state, params)
  A, b = params
  return A * tanh.(state) .+ b
end

fixed_point_wrapper(f, x, p) = fixed_point(f, x, p; verbose=true)

function loss(params)
  x_star = fixed_point_wrapper(f, x_init, params)
  return abs(sum(x_star))
end


begin
  # 参数
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1 0.2; 0.1 0.2]
  params = (A, b)
  state_init = ones(Float64, 2, 2)

  # 计算固定点
  state_star = fixed_point_wrapper(f, state_init, params)
  display(state_star)
end

# f(state_star, params)
# @run 
# dparam = make_zero(params)
# dparam[1] = 1

f_self = (p) -> fixed_point(f, state_init, p)  # 看的是p -> x_star的影响
f_orgi = (p) -> _fixed_point(f, state_init, p) #

funcs = [f_self, f_orgi]
res = []


dparam = make_zero(params)
dparam[1][1] = 1.0
dparam[2][1] = 1.0
# 

map(f -> begin
    result = Enzyme.autodiff(
      ForwardWithPrimal, f,
      Duplicated(params, dparam)
    )
    grad = result[2]
end, funcs)
