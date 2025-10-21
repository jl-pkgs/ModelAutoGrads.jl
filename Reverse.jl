# ============================================================================
# Enzyme 反向模式自定义规则 (Reverse Mode / VJP)
# ============================================================================

"""
Augmented primal: 前向传播 + 保存反向传播需要的数据

保存到 tape:
- state*: 收敛的固定点
- J_state: ∂f/∂state|(state*, param)
- param_val: param 的值
- args_vals: args 的值
- 其他元信息
"""
function Enzyme.EnzymeRules.augmented_primal(
    config::RevConfigWidth{1},
    func::Const{typeof(fixed_point)},
    ::Type{<:Duplicated},
    f::Const,
    state::Duplicated,
    param::Duplicated,
    args::Const...;
    kwargs...
)
    
    if get(kwargs, :verbose, false)
        println("使用自定义反向模式规则 (augmented primal)")
    end
    
    # 1. 前向传播：计算固定点
    state_star = fixed_point(f.val, state.val, param.val, map(a -> a.val, args)...; kwargs...)
    
    # 处理标量情况
    state_is_scalar = !(state_star isa AbstractArray)
    if state_is_scalar
        state_star_vec = [state_star]
    else
        state_star_vec = state_star
    end
    n = length(state_star_vec)
    
    # 2. 提取传递给 f 的关键字参数
    f_kwargs = filter(kw -> kw.first ∉ [:tol, :max_iters, :norm_type, :verbose], pairs(kwargs))
    
    # 3. 计算并保存 J_state = ∂f/∂state|(state*, param)
    J_state = zeros(n, n)
    
    for i in 1:n
        dstate = zeros(n)
        dstate[i] = 1.0
        
        state_dup = state_is_scalar ? Duplicated(state_star, dstate[1]) : Duplicated(copy(state_star_vec), dstate)
        param_const = Const(param.val)
        args_const = map(Const, map(a -> a.val, args))
        
        if state_is_scalar
            result = autodiff(
                Forward,
                f.val,
                Duplicated,
                state_dup,
                param_const,
                args_const...
            )
            J_state[:, i] = [result.dval]
        else
            result_val = similar(state_star_vec)
            result_dval = zeros(n)
            autodiff(
                Forward,
                (out, s, p, a...) -> (out .= f.val(s, p, a...; f_kwargs...); nothing),
                Const,
                Duplicated(result_val, result_dval),
                state_dup,
                param_const,
                args_const...
            )
            J_state[:, i] = result_dval
        end
    end
    
    # 4. 保存数据到 tape
    tape = (
        state_star = state_star_vec,
        J_state = J_state,
        param_val = param.val,
        args_vals = map(a -> a.val, args),
        f_func = f.val,
        f_kwargs = f_kwargs,
        state_is_scalar = state_is_scalar
    )
    
    # 5. 返回
    primal_return = state_is_scalar ? state_star : copy(state_star_vec)
    shadow_return = state_is_scalar ? zero(state_star) : zeros(n)
    
    return Enzyme.EnzymeRules.AugmentedReturn(
        Duplicated(primal_return, shadow_return),
        nothing,
        tape
    )
end

"""
Reverse pass: 反向传播梯度

数学原理:
给定上游梯度 dL/dstate*，计算 dL/dparam

伴随方程:
(I - J_state)^T · λ = dL/dstate*

然后:
dL/dparam = λ^T · J_param = λ^T · ∂f/∂param|(state*, param)
"""
function Enzyme.EnzymeRules.reverse(
    config::RevConfigWidth{1},
    func::Const{typeof(fixed_point)},
    dret::Duplicated,
    tape,
    f::Const,
    state::Duplicated,
    param::Duplicated,
    args::Const...;
    kwargs...
)
    
    if get(kwargs, :verbose, false)
        println("使用自定义反向模式规则 (reverse pass)")
    end
    
    # 1. 从 tape 恢复数据
    state_star = tape.state_star
    J_state = tape.J_state
    param_val = tape.param_val
    args_vals = tape.args_vals
    f_func = tape.f_func
    f_kwargs = tape.f_kwargs
    state_is_scalar = tape.state_is_scalar
    
    n = length(state_star)
    
    # 2. 获取上游梯度 dL/dstate*
    dL_dstate_star = state_is_scalar ? [dret.dval] : dret.dval
    
    # 3. 求解伴随方程: (I - J_state)^T · λ = dL/dstate*
    A_T = (I - J_state)'
    lambda = A_T \ dL_dstate_star
    
    # 4. 计算 dL/dparam = λ^T · J_param
    # 使用反向模式计算 J_param^T · λ
    
    # 构造一个辅助函数来计算 J_param 的列
    param_is_scalar = !(param_val isa AbstractArray)
    m = param_is_scalar ? 1 : length(param_val)
    
    param_grad = zeros(m)
    
    for i in 1:m
        dparam = param_is_scalar ? 1.0 : (p = zeros(m); p[i] = 1.0; p)
        
        param_dup = param_is_scalar ? Duplicated(param_val, dparam) : Duplicated(copy(param_val), dparam)
        state_const = state_is_scalar ? Const(state_star[1]) : Const(state_star)
        args_const = map(Const, args_vals)
        
        if state_is_scalar
            result = autodiff(
                Forward,
                f_func,
                Duplicated,
                state_const,
                param_dup,
                args_const...
            )
            J_param_col = [result.dval]
        else
            result_val = similar(state_star)
            result_dval = zeros(n)
            autodiff(
                Forward,
                (out, s, p, a...) -> (out .= f_func(s, p, a...; f_kwargs...); nothing),
                Const,
                Duplicated(result_val, result_dval),
                state_const,
                param_dup,
                args_const...
            )
            J_param_col = result_dval
        end
        
        # λ^T · J_param[:, i]
        param_grad[i] = dot(lambda, J_param_col)
    end
    
    # 5. 累加梯度到 param.dval
    if param_is_scalar
        param.dval += param_grad[1]
    else
        param.dval .+= param_grad
    end
    
    # 6. 返回 (对 state 的梯度不需要，因为它是迭代的中间变量)
    return (nothing, nothing, nothing, map(_ -> nothing, args)...)
end

