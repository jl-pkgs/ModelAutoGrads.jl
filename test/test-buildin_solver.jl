@testset "[build-in] solver" begin
  # 使用不同的初始状态来区分各元素的贡献
  A = [0.7 0.2; 0.1 0.6]
  b = [0.1 0.7; 0.5 0.4]
  params = ComponentArray(; A, b)

  state_init = [1.0 2.0; 3.0 4.0]  # 不同的值
  state_star = zeros(Float64, 2, 2)

  m = length(state_init)
  n = length(params)

  grad_f = gradient_forward(f, state_init, params)
  grad_f2 = gradient_forward!(f!, state_init, params)

  grad_r = gradient_reverse!(f!, state_init, params)

  @test grad_r ≈ grad_f
  @test grad_r ≈ grad_f2
  # res = gradient(Forward, params -> f(state_init, params), params)
  # grad4 = reshape(res[1], m, n)
  # @test grad_r ≈ grad4
end

# Jacobian 矩阵：
# 4×8 Matrix{Float64}:
#  0.761594  0.0       0.995055  0.0       1.0  0.0  0.0  0.0
#  0.0       0.761594  0.0       0.995055  0.0  1.0  0.0  0.0
#  0.964028  0.0       0.999329  0.0       0.0  0.0  1.0  0.0
#  0.0       0.964028  0.0       0.999329  0.0  0.0  0.0  1.0
