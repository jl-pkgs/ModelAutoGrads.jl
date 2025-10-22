using ModelAutoGrad, Enzyme, Test
using ComponentArrays
# include("main_func.jl")

# grad_f = gradient_forward(f, state_init, params)

function gradient_reverse_fixed(f!, state_init, params; solver=fixed_point!)
  m = length(state_init)
  n = length(params)
  J = zeros(m, n) # m, n

  for i in 1:m
    d_res = make_zero(state_init)
    d_res[i] = 1.0

    d_param = make_zero(params)
    d_st = make_zero(state_init)

    Enzyme.autodiff(
      Reverse,
      solver,
      Const,
      Duplicated(make_zero(state_init), d_res), # 输入
      Const(f!),
      Const(state_init),
      Duplicated(params, d_param)               # 输出
    )
    J[i, :] = d_param[:]
  end
  J
end


# function f!(state_next, state, params, args...; kw...)
#   # A, b = params.A, params.b
#   (; A, b) = params
#   R = A * tanh.(state) .+ b # 创建一个临时变量
#   copyto!(state_next, R)
#   return nothing
# end


@testset "[fixed_point] gradient_reverse" begin
  # 使用不同的初始状态来区分各元素的贡献
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1 0.7; 0.5 0.4]
  params = ComponentArray(; A, b)

  state_init = [1.0 2.0; 3.0 4.0]  # 不同的值

  @time J_custom = gradient_reverse_fixed(f!, state_init, params; solver=fixed_point!)
  @time J = gradient_reverse_fixed(f!, state_init, params; solver=_fixed_point!) # 对照组

  @test maximum(abs.(J_custom - J)) <= 0.5 * 1e-4
end
