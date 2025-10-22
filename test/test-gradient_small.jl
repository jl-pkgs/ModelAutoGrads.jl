using ModelAutoGrad, Enzyme, ComponentArrays, Test


function test_reverse(f!, state_init, params; solver=fixed_point!)
  d_res = make_zero(state_init)
  d_res[1] = 1.0

  d_param = make_zero(params)
  d_st = make_zero(state_init)

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


function test_forward(f, state_init, params; solver=fixed_point)
  d_param = make_zero(params)
  d_param[1] = 1.0
  # res = make_zero(state_init)
  # d_res = make_zero(state_init)

  grads, _ = autodiff(
    ForwardWithPrimal, solver,
    # Duplicated(res, d_res),
    Const(f),
    Const(state_init),
    Duplicated(params, d_param)
  )
  d_param
end


## 基于最小单元进行测试
@testset "[fixed_point] rev/fwd smallest pieces" begin
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1 0.7; 0.5 0.4]
  params = ComponentArray(; A, b)

  state_init = [1.0 2.0; 3.0 4.0]  # 不同的值
  # state_star = make_zero(state_init)

  dp = test_reverse(f!, state_init, params; solver=fixed_point!)
  _dp = test_reverse(f!, state_init, params; solver=_fixed_point!)
  @test maximum(abs.(dp - _dp)) <= 1e-4

  dp = test_forward(f, state_init, params; solver=fixed_point)
  _dp = test_forward(f, state_init, params; solver=_fixed_point)
  @test maximum(abs.(dp - _dp)) <= 1e-4
end
