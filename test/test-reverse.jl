# using Pkg
# Pkg.activate(".")

using ModelAutoGrad
using Enzyme
using Enzyme: autodiff


f(x, params) = params[1] * tanh(x) + params[2]

# 创建包装函数，不带 verbose 参数以避免 kwargs 问题
fixed_point_wrapper(f, x, p) = fixed_point(f, x, p; verbose=true)


function loss(params)
  x_star = fixed_point_wrapper(f, x_init, params)
  return x_star^2
end


begin
  params = [0.8, 0.2]
  x_init = 2.0

  # 前向模式：计算 dx*/dparams[1]
  println("\n前向模式测试:")

  # 在前向模式中，使用 Duplicated 包装参数和返回值
  # 输入的 dval 字段包含切向量（tangent vector）
  # 输出的 dval 字段将包含结果的导数
  dps = make_zero(params)
  dps[1] = 1.0

  # 使用 ForwardWithPrimal 模式返回原始值和切向量
  result = Enzyme.autodiff(
    ForwardWithPrimal,
    (p) -> fixed_point_wrapper(f, x_init, p), ## 看的是p -> x_star的影响
    Duplicated(params, dps)
  )
  grad = result[1]
  @test grad ≈ 1.2356672989421054
end


begin
  # 反向模式：计算梯度
  println("\n反向模式测试:")

  x_tangent = result[1]  # 导数/切向量
  x_primal = result[2]   # 原始值

  println("固定点 x*: ", x_primal)
  println("dx*/dparams[1]: ", x_tangent)

  ## 两种求梯度的方法，结果一致
  dps = make_zero(params) # 计算之前，必须要归零
  autodiff(Reverse, (p) -> fixed_point_wrapper(f, x_init, p),
    Active,
    Duplicated(params, dps)
  )
  isapprox(dps[1], grad; atol=1e-5)
end

begin
  dps = make_zero(params)
  autodiff(Reverse, loss, Active, Duplicated(params, dps))
  
  println("参数: ", params)
  println("dL/dparams: ", dps)
end

