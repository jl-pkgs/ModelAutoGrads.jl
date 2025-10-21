
"""
示例 4: 数值验证梯度
"""
# function example_gradient_check()
begin
  println("\n" * "="^60)
  println("示例 4: 数值梯度验证")
  println("="^60)

  f(x, params) = params[1] * tanh(x) + params[2]

  params = [0.8, 0.2]
  x_init = 0.0

  # 定义损失函数
  function loss(p)
    x_star = fixed_point(f, x_init, p; verbose=false)
    return x_star^2
  end

  # 解析梯度（使用 Enzyme）
  params_dup = Duplicated(copy(params), zeros(2))
  autodiff(Reverse, loss, Active, params_dup)
  grad_analytical = copy(params_dup.dval)

  # 数值梯度（有限差分）
  epsilon = 1e-7
  grad_numerical = zeros(2)
  for i in 1:2
    params_plus = copy(params)
    params_plus[i] += epsilon
    params_minus = copy(params)
    params_minus[i] -= epsilon

    grad_numerical[i] = (loss(params_plus) - loss(params_minus)) / (2 * epsilon)
  end

  println("解析梯度: ", grad_analytical)
  println("数值梯度: ", grad_numerical)
  println("相对误差: ", norm(grad_analytical - grad_numerical) / norm(grad_numerical))
end
