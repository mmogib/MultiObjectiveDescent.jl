using MultiObjectiveDescent
using Test

@testset "MultiObjectiveDescent.jl" begin
    include("test_types.jl")
    include("test_subproblem.jl")
end
