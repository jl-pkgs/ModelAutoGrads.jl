using ModelAutoGrad, Enzyme, ComponentArrays, Test


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



include("test-buildin_solver.jl")
include("test-gradient.jl")
include("test-gradient_small.jl")


@testset "fixed_point" begin
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1 0.7; 0.5 0.4]
  params = ComponentArray(; A, b)

  state_init = [1.0 2.0; 3.0 4.0]  # 不同的值
  state_star = make_zero(state_init)

  r1 = fixed_point(f, state_init, params)
  r2 = _fixed_point(f, state_init, params)
  @test r1 == r2

  s1 = deepcopy(r1)
  s2 = deepcopy(r2)

  fixed_point!(s1, f!, state_init, params)
  _fixed_point!(s2, f!, state_init, params)
  @test s1 == s2
  @test s1 == r1
end
