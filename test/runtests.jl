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
    G = Gop(PSFs, objL, imgL, input_channels, output_channels)

    @test size(G) == (imgL^2 * output_channels, objL^2 * input_channels)

    m, n = size(G)
    u = rand(n)
    v = rand(m)
    vtmp = similar(v)
    utmp = similar(u)
    # Compare accuracy of output to expected
    out_expected = zeros(imgL, imgL, output_channels)
    for i in 1:input_channels
        for j in 1:output_channels
            input_slice = reshape(u, objL, objL, input_channels)[:, :, i]
            kernel = PSFs[:, :, i, j]
            out_expected[:, :, j] += real.(convolve(input_slice, fft(kernel)))
        end
    end
    out_actual = reshape(G * u, imgL, imgL, output_channels)
    # @test size(out ≈ out_actual
            
    # Test consistency of out-of-place and in-place 
    mul!(vtmp, G, u)
    mul!(utmp, G', v)
    @test vtmp ≈ G * u
    @test utmp ≈ G' * v
    # Dot product test for transposition
    @test dot(v, G*u) ≈ dot(u, G'*v)
end
