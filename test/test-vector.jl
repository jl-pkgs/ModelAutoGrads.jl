using ModelAutoGrad
using Enzyme, Test
using Enzyme: autodiff
using ComponentArrays


# 定义迭代函数
function f(state, params)
  A, b = params
  R = A * tanh.(state) .+ b
  state[1] = R[1] ## 迭代过程中state被修改
  return R
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
  params = (; A, b)

  state_init = ones(Float64, 2, 2)

  # 计算固定点
  state_star = fixed_point_wrapper(f, state_init, params)
  display(state_star)
end


@testset "Forward fixed_point works" begin
  solvers = [fixed_point, _fixed_point]

  ## 如何更新梯度
  dparam = make_zero(params)
  dparam[1][1] = 1.0
  dparam[2][1] = 1.0

  res = map(solver -> begin
      state_init = ones(Float64, 2, 2)

      result = Enzyme.autodiff(
        ForwardWithPrimal, solver,
        Const(f),
        Const(state_init),
        Duplicated(params, dparam)
      )
      grad = result[2]
    end, solvers)

  @test maximum(abs.(res[1] - res[2])) <= 1e-6
end


# 采用Functors把models切分为
# params, layer = 
# params构建为ComponentArrays
# ps = ComponentArray(params)
# dps = make_zero(ps)
# dps[2]
# length(dps) # 这次返回的是正确的

# f(state_star, params)
# @run 
# dparam = make_zero(params)
# dparam[1] = 1

# f_self = (p) -> fixed_point(f, state_init, p)  # 看的是p -> x_star的影响
# f_orgi = (p) -> _fixed_point(f, state_init, p) #
# funcs = [f_self, f_orgi]
# res = []
