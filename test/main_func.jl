using ComponentArrays, Enzyme


function gradient_forward(f, state_init, params)
  n = length(params)
  m = length(state_star)
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
  m = length(state_star)
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

  m = length(state_init)
  n = length(params)
end


# ## test pdv(F, state)
# d_st = make_zero(state_init)

# autodiff(
#   Forward,
#   f,
#   # Duplicated(make_zero(state_init), d_st), # 输出
#   Duplicated(state_init, d_st),
#   Const(params)
# )

# autodiff(
#   Forward,
#   f!,
#   Const,
#   Duplicated(make_zero(state_init), d_st), # 输出
#   Duplicated(state_init, d_st),
#   Const(params)
# )
