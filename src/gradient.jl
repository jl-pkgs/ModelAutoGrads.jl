export gradient_reverse_fixed, gradient_forward_fixed


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
