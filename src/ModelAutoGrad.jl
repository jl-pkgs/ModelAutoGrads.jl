module ModelAutoGrad


export norm
export reverse, forward

# using Enzyme.EnzymeRules
using Enzyme
import Enzyme: autodiff
import Enzyme.EnzymeRules
import Enzyme.EnzymeRules: augmented_primal, reverse, forward

using LinearAlgebra
# greet() = print("Hello World!")

include("fixed_point.jl")
include("Forward.jl")
include("Reverse.jl")

end # module ModelAutoGrad
