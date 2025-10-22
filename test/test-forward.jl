using ModelAutoGrad, Enzyme, Test
using ComponentArrays
# include("main_func.jl")

function f(state, params, args...; kw...)
  # A, b = params.A, params.b
  (; A, b) = params
  return A * tanh.(state) .+ b
end

function f!(state_next, state, params, args...; kw...)
  # A, b = params.A, params.b
  (; A, b) = params
  R = A * tanh.(state) .+ b # 创建一个临时变量
  copyto!(state_next, R)
  return nothing
end


begin
  # 使用不同的初始状态来区分各元素的贡献
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1 0.7; 0.5 0.4]
  params = ComponentArray(; A, b)

  state_init = [1.0 2.0; 3.0 4.0]  # 不同的值
  state_star = make_zero(state_init)
  # m = length(state_init)
  # n = length(params)
end


# grad_f = gradient_forward(f, state_init, params)
function gradient_forward_fixed(f, state_init, params; solver=fixed_point!)
  n = length(params)
  m = length(state_init)

  J = zeros(m, n)
  for i in 1:n
    d_param = make_zero(params)
    d_param[i] = 1.0

    res = make_zero(state_init)
    d_res = make_zero(state_init)

    grads, _ = Enzyme.autodiff(
      ForwardWithPrimal, solver,
      # Duplicated(res, d_res),
      Const(f),
      Const(state_init),
      Duplicated(params, d_param)
    )
    J[:, i] = grads[:]
  end
  J
end

@testset "[fixed_point] gradient_forward" begin
  @time J_custom = gradient_forward_fixed(f, state_init, params; solver=fixed_point)
  @time J = gradient_forward_fixed(f, state_init, params; solver=_fixed_point)

  @test maximum(abs.(J_custom - J)) <= 0.5 * 1e-4
end

