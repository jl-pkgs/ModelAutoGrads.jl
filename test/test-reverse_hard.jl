using ModelAutoGrad, ComponentArrays, Test
using Enzyme, Enzyme.EnzymeRules

# function f(state, params, args...; kw...)
#   # A, b = params.A, params.b
#   (; A, b) = params
#   return A * tanh.(state) .+ b
# end

function f!(state_next, state, params, args...; kw...)
  # A, b = params.A, params.b
  (; A, b) = params
  R = A * tanh.(state) .+ b # 创建一个临时变量
  copyto!(state_next, R)
  return nothing
end

function test_reverse_high(f!, state_init, params; solver=fixed_point!)
  d_res = make_zero(state_init)
  d_res[1] = 1.0

  d_param = make_zero(params)
  # d_st = make_zero(state_init)
  autodiff(
    Reverse,
    solver,
    Const,
    Duplicated(make_zero(state_init), d_res), # 输入
    Const(f!),
    Const(state_init),
    Duplicated(params, d_param)               # 输出
  )
  d_param
end


function test_reverse_hard(f!, state_init, params)
  d_res = make_zero(state_init)
  d_res[1] = 1.0

  d_param = make_zero(params)
  # d_st = make_zero(state_init)

  config = RevConfigWidth{1,false,false,(false, false, false, false, false),false,false}()
  func = Const{typeof(fixed_point!)}(ModelAutoGrad.fixed_point!)

  primal = EnzymeRules.augmented_primal(
    config,
    func,
    Const{Nothing},
    Duplicated(make_zero(state_init), d_res), # 输入
    Const(f!),
    Const(state_init),
    Duplicated(params, d_param)               # 输出
  )
  tape = primal.tape

  EnzymeRules.reverse(
    config,
    func,
    Const{Nothing},
    tape,
    Duplicated(make_zero(state_init), d_res), # 输入
    Const(f!),
    Const(state_init),
    Duplicated(params, d_param)               # 输出
  )
  d_param
end

@testset "test reverse_hard" begin
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1 0.7; 0.5 0.4]
  params = ComponentArray(; A, b)

  state_init = [1.0 2.0; 3.0 4.0]  # 不同的值
  state_star = make_zero(state_init)

  dp1 = test_reverse_hard(f!, state_init, params)
  dp2 = test_reverse_high(f!, state_init, params)
  @test dp1 ≈ dp2
  # r1 = fixed_point(f, state_init, params)
  # r2 = _fixed_point(f, state_init, params)
  # @test r1 == r2
end
