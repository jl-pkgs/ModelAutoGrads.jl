"""
示例 3: 带额外参数和关键字参数
"""
# function example_with_args()
begin
  println("\n" * "="^60)
  println("示例 3: 带额外参数")
  println("="^60)

  # 迭代函数带额外参数
  function f(x, params, scale; offset=0.0)
    return params[1] * tanh(scale * x) .+ params[2] .+ offset
  end

  params = [0.8, 0.2]
  x_init = 0.0
  scale = 2.0  # 额外位置参数

  # 不带 offset
  x_star1 = fixed_point(f, x_init, params, scale; verbose=false)
  println("固定点 (offset=0): ", x_star1)

  # 带 offset
  x_star2 = fixed_point(f, x_init, params, scale; offset=0.1, verbose=false)
  println("固定点 (offset=0.1): ", x_star2)
end
