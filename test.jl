include("fixed_point.jl")

# ============================================================================
# 测试和示例
# ============================================================================

"""
示例 1: 简单的标量固定点
x* = a * tanh(x*) + b
"""
function example_scalar()
    println("\n" * "="^60)
    println("示例 1: 标量固定点")
    println("="^60)
    
    # 定义迭代函数
    f(x, params) = params[1] * tanh(x) + params[2]
    
    # 参数
    params = [0.8, 0.2]
    x_init = 0.0
    
    # 前向模式：计算 dx*/dparams[1]
    println("\n前向模式测试:")
    params_dup = Duplicated(copy(params), [1.0, 0.0])
    x_dup = Duplicated(x_init, 0.0)
    
    result = autodiff(
        Forward,
        fixed_point,
        Duplicated,
        Const(f),
        x_dup,
        params_dup;
        verbose=false
    )
    
    println("固定点 x*: ", result.val)
    println("dx*/dparams[1]: ", result.dval)
    
    # 反向模式：计算梯度
    println("\n反向模式测试:")
    
    function loss(params)
        x_star = fixed_point(f, x_init, params; verbose=false)
        return x_star^2
    end
    
    params_dup = Duplicated(copy(params), zeros(2))
    autodiff(Reverse, loss, Active, params_dup)
    
    println("参数: ", params)
    println("dL/dparams: ", params_dup.dval)
end

"""
示例 2: 向量固定点
state* = A * tanh(state*) + b
"""
function example_vector()
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

"""
示例 3: 带额外参数和关键字参数
"""
function example_with_args()
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

"""
示例 4: 数值验证梯度
"""
function example_gradient_check()
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

# 运行所有示例
begin
    example_scalar()
    # example_vector()
    # example_with_args()
    # example_gradient_check()
    
    # println("\n" * "="^60)
    # println("所有测试完成！")
    # println("="^60)
end
