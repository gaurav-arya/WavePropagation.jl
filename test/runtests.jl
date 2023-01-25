using Test
using WavePropagation
using FFTW
using LinearAlgebra

@testset "Forward model" begin
    # Create a test G
    input_channels = 5
    output_channels = 3
    objL = 20
    imgL = 10
    PSFs = rand(objL + imgL, objL + imgL, input_channels, output_channels)
    fftPSFs = mapslices(fft, PSFs; dims=(1,2))
    G = Gop(fftPSFs, objL, imgL, input_channels, output_channels)

    @test size(G) == (imgL^2 * output_channels, objL^2 * input_channels)

    # Test consistency of out-of-place and in-place, and of transposition
    m, n = size(G)
    u = rand(n)
    v = rand(m)
    vtmp = similar(v)
    utmp = similar(u)
    mul!(vtmp, G, u)
    mul!(utmp, G', v)
    @test vtmp ≈ G * u
    @test utmp ≈ G' * v
    @test dot(v, G*u) ≈ dot(u, G'*v)
end
