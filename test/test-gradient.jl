using ModelAutoGrad, Enzyme, ComponentArrays, Test
# include("main_func.jl")

# function f!(state_next, state, params, args...; kw...)
#   # A, b = params.A, params.b
#   (; A, b) = params
#   R = A * tanh.(state) .+ b # 创建一个临时变量
#   copyto!(state_next, R)
#   return nothing
# end

# function f(state, params, args...; kw...)
#   # A, b = params.A, params.b
#   (; A, b) = params
#   return A * tanh.(state) .+ b
# end

# grad_f = gradient_forward(f, state_init, params)

@testset "[fixed_point] gradient_forward" begin
  # 使用不同的初始状态来区分各元素的贡献
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1 0.7; 0.5 0.4]
  params = ComponentArray(; A, b)

  state_init = [1.0 2.0; 3.0 4.0]  # 不同的值
  # state_star = make_zero(state_init)

  @time J = gradient_forward_fixed(f, state_init, params; solver=_fixed_point)
  @time J_custom = gradient_forward_fixed(f, state_init, params; solver=fixed_point)

  @test maximum(abs.(J_custom - J)) <= 0.5 * 1e-4
end


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

