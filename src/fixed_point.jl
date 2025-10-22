## 对照组
# https://enzyme.mit.edu/julia/dev/api/
export _fixed_point, _fixed_point!


function _fixed_point!(
  state_curr, f!, state, param, args...;
  tol::Float64=1e-6, nmax::Int=1000, norm_type::Real=2, verbose::Bool=false, kw...)

  state_prev = copy(state)
  f!(state_curr, state, param, args...; kw...) # 第一次迭代：state_curr = f(state)

  for iter in 1:nmax
    residual = norm(state_curr - state_prev, norm_type)

    if residual < tol
      return nothing
    end

    state_prev .= state_curr # update
    f!(state_curr, state_prev, param, args...; kw...)
  end
  return nothing
end


function _fixed_point(f, state, param, args...;
  tol::Float64=1e-6, nmax::Int=1000, norm_type::Real=2, kw...)

  state_prev = copy(state)
  state_curr = f(state, param, args...; kw...)

  for iter in 1:nmax
    residual = norm(state_curr - state_prev, norm_type)
    if residual < tol
      return state_curr
    end

    state_prev = copy(state_curr)
    state_curr = f(state_curr, param, args...; kw...)
  end
  return state_curr
end
