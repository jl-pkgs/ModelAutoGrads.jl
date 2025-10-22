using ModelAutoGrad, Enzyme, Test
using ComponentArrays
include("main_func.jl")

grad_f = gradient_forward(f, state_init, params)

function gradient_reverse_fixed(f!, state_init, params; solver = fixed_point!)
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


# using BenchmarkTools
# custom
@time J_custom = gradient_reverse_fixed(f!, state_init, params; solver=fixed_point!)

# 对照组
@time J_corr = gradient_reverse_fixed(f!, state_init, params; solver=_fixed_point!)
