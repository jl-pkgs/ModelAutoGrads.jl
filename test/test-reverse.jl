using ModelAutoGrad
using Enzyme, Test

include("main_funcs.jl")

begin
  d_param = make_zero(params)
  d_param.A[1] = 1.0

  grads, zval = Enzyme.autodiff(
    ForwardWithPrimal, fixed_point,
    Const(f),
    Const(state_init),
    Duplicated(params, d_param)
  )
  grads
end


# gradient
function cal_gradient()
  m = length(state_star)
  n = length(params)
  J = zeros(m, n) # m, n

  res = copy(state_init)

  for i in 1:m
    res_dval = make_zero(state_star)
    res_dval[i] = 1.0

    d_param = make_zero(params)
    d_st = make_zero(state_init)

    result = Enzyme.autodiff(
      Reverse,
      fixed_point!,
      Const,
      Duplicated(res, res_dval), # 返回值
      Const(f!),
      # Const(state_init),
      Duplicated(state_init, d_st),
      Duplicated(params, d_param)
    )
    
    J[i, :] = d_param[:]
  end
  J
end

cal_gradient()

# fixed_point!(res, f!, state_init, params)
