using ModelAutoGrad, Enzyme, Test
using ComponentArrays
include("main_func.jl")

grad_f = gradient_forward(f, state_init, params)

function gradient_forward_fixed(f, state_init, params; solver=fixed_point!)
  n = length(params)
  m = length(state_init)

  J = zeros(m, n)
  for i in 1:n
    d_param = make_zero(params)
    d_param[i] = 1.0

    grads = Enzyme.autodiff(
      Forward, solver,
      Const(f),
      Const(state_init),
      Duplicated(params, d_param)
    )[1]
    J[:, i] = grads[:]
  end
  J
end

@time J_custom = gradient_forward_fixed(f, state_init, params; solver=fixed_point)

@time J = gradient_forward_fixed(f, state_init, params; solver=_fixed_point)



# 计算固定点
# fixed_point!(state_star, f, state_init, params)
# display(state_star)

@testset "Forward fixed_point works" begin
  solvers = [fixed_point, _fixed_point]

  ## 如何更新梯度
  d_param = make_zero(params)
  d_param.A[1] = 1.0
  d_param.b[1] = 1.0

  res = map(solver -> begin
      result = Enzyme.autodiff(
        ForwardWithPrimal, solver,
        Const(f),
        Const(state_init),
        Duplicated(params, d_param)
      )
      @show result

      grad = result[2]
    end, solvers)
    
  @test maximum(abs.(res[1] - res[2])) <= 1e-6
end


