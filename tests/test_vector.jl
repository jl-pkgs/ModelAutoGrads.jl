"""
示例 2: 向量固定点
state* = A * tanh(state*) + b
"""

# function example_vector()
begin
  println("\n" * "="^60)
  println("示例 2: 向量固定点")
  println("="^60)

  # 定义迭代函数
  function f(state, params)
    A, b = params
    return A * tanh.(state) .+ b
  end

  # 参数
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1, 0.2]
  params = (A, b)
  state_init = zeros(2)

  # 计算固定点
  println("\n计算固定点:")
  state_star = fixed_point(f, state_init, params; verbose=false)
  println("state*: ", state_star)

  # 验证: state* ≈ f(state*, params)
  residual = norm(state_star - f(state_star, params))
  println("残差: ", residual)
end
