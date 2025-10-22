using ModelAutoGrad
using Enzyme, Test
using ComponentArrays


# 定义迭代函数
function f(state, params, args...; kw...)
  A, b = params
  R = A * tanh.(state) .+ b
  # state[1] = R[1] ## 迭代过程中state被修改
  return R
end

# function loss(params)
#   x_star = fixed_point_wrapper(f, x_init, params)
#   return abs(sum(x_star))
# end

begin
  # 参数
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1 0.2; 0.1 0.2]
  params = (; A, b)

  state_init = ones(Float64, 2, 2)

  # 计算固定点
  state_star = fixed_point(f, state_init, params)
  display(state_star)

  res = make_zero(state_star)
  res_dval = make_zero(state_star)
  res_dval[1] = 1.0
end

# f(state_init, params)


# @testset "Forward fixed_point works" 
begin
  # solvers = [fixed_point, _fixed_point]
  ## 如何更新梯度
  dparam = make_zero(params)
  # dparam[1][1] = 1.0
  # dparam[2][1] = 1.0
  state_init = ones(Float64, 2, 2)

  result = Enzyme.autodiff(
    Reverse, 
    fixed_point!,
    Const,
    Duplicated(res, res_dval), # 返回值
    Const(f),
    # Const(state_init),
    Duplicated(state_init, make_zero(state_init)),
    Duplicated(params, dparam)
  )
end

f(state_init, params)

fixed_point(f, state_init, params)


# grad = result[2]

# res = map(solver -> begin
#     state_init = ones(Float64, 2, 2)

#     result = Enzyme.autodiff(
#       ForwardWithPrimal, solver,
#       Const(f),
#       Const(state_init),
#       Duplicated(params, dparam)
#     )
#     grad = result[2]
#   end, solvers)

# @test maximum(abs.(res[1] - res[2])) <= 1e-6
