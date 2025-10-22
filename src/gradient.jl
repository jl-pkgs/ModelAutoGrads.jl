export gradient_forward, gradient_forward!, gradient_reverse!


function gradient_forward(f, state_init, params)
  n = length(params)
  m = length(state_init)
  J = zeros(m, n)

  for i in 1:n
    d_param = make_zero(params)
    d_param[i] = 1.0

    grads = autodiff(
      Forward,
      f,
      Const(state_init),
      Duplicated(params, d_param)
    )[1]
    J[:, i] = vec(grads)
  end
  J
end


function gradient_forward!(f!, state_init, params)
  n = length(params)
  m = length(state_init)
  J = zeros(m, n)

  for i in 1:n
    d_param = make_zero(params)
    d_param[i] = 1.0
    d_st = make_zero(state_init)

    autodiff(
      Forward,
      f!,
      Const,
      Duplicated(make_zero(state_init), d_st), # 输出
      Const(state_init),
      Duplicated(params, d_param)
    )
    J[:, i] = vec(d_st)
  end
  J
end

function gradient_reverse!(f!, state_init, params)
  m = length(state_init)
  n = length(params)
  J = zeros(m, n)

  for i in 1:m
    d_st = zeros(size(state_init))
    d_st[i] = 1.0
    d_param = make_zero(params)

    autodiff(
      Reverse, f!, Const,
      Duplicated(make_zero(state_init), d_st), # 输入
      Const(state_init),
      Duplicated(params, d_param)              # 输出
    )
    J[i, :] = vec(d_param)
  end
  return J
end
