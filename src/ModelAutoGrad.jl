module ModelAutoGrad


export norm
export reverse, forward, augmented_primal

export fixed_point, fixed_point!
export _fixed_point, _fixed_point!

# using Enzyme.EnzymeRules
using Enzyme
using Enzyme.EnzymeRules
import Enzyme: autodiff
import Enzyme.EnzymeRules
import Enzyme.EnzymeRules: augmented_primal, reverse, forward

using LinearAlgebra


include("fixed_point.jl")

include("Forward.jl")
include("Reverse.jl")

include("gradient.jl")


end # module ModelAutoGrad
