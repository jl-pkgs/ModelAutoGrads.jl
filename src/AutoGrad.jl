module AutoGrad


export norm


using Enzyme
using Enzyme.EnzymeRules
using LinearAlgebra
# greet() = print("Hello World!")

include("fixed_point.jl")
include("Forward.jl")


end # module AutoGrad
