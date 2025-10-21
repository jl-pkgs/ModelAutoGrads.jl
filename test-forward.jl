using Pkg
Pkg.activate(".")

using AutoGrad
using Enzyme
using Enzyme: autodiff


f(x, params) = params[1] * tanh(x) + params[2]

# 创建包装函数，固定 verbose 参数
fixed_point_wrapper(f, x, p) = fixed_point(f, x, p; verbose=true)

function loss(params)
  x_star = fixed_point(f, x_init, params; verbose=false)
  return x_star^2
end


# 示例 1: 简单的标量固定点
# x* = a * tanh(x*) + b
begin
  params = [0.8, 0.2]
  x_init = 0.0

  # 前向模式：计算 dx*/dparams[1]
  println("\n前向模式测试:")

  # 在前向模式中，使用 Duplicated 包装参数和返回值
  # 输入的 dval 字段包含切向量（tangent vector）
  # 输出的 dval 字段将包含结果的导数
  params_dup = Duplicated(copy(params), [1.0, 0.0])

  # 使用 ForwardWithPrimal 模式返回原始值和切向量
  result = Enzyme.autodiff(
    ForwardWithPrimal,
    (p) -> fixed_point_wrapper(f, x_init, p),
    # Duplicated,
    params_dup
  )
end
