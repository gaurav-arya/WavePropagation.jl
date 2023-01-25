module WavePropagation
    
    using FFTW
    using FastChebInterp
    using FastChebInterp: ChebPoly
    using Memoize
    using LinearAlgebra
    using ChainRulesCore: ProjectTo, NoTangent
    using ThreadsX
    import ChainRulesCore.rrule

    export planned_fft, planned_ifft
    export convolve, convolveT, convolve!, convolveT!
    export incident_field, greens
    export get_model
    export Gop

    include("planned_fft.jl")
    include("fields.jl")
    include("convolve.jl")
    include("scattering.jl")
    include("forward_model.jl")

end
