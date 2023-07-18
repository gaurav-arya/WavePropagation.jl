using LinearMaps

struct Gop{AA1, AA2} <: LinearMap{Float64}
    fftPSFs::AA1 # todo: fix
    objL::Int
    imgL::Int
    input_channels::Int 
    output_channels::Int
    padded::AA2
end   

"""
    Gop(PSFs, objL, imgL, input_channels, output_channels)

Create a convolutional linear operator with a specified number of input and output channels. 
The operator takes an input of shape `(objL, objL, input_channels)`
and gives output of shape `(imgL, imgL, output_channels)`.
The provided `PSFs` is expected to have shape `(psfL, psfL, input_channels, output_channels)`,
where `psfL = objL + imgL`.
Note that the operator expepcts flattened input and output.
"""
function Gop(PSFs, objL, imgL, input_channels, output_channels)
    psfL = objL + imgL
    fftPSFs = stack([planned_fft(PSFs[:, :, i_input, i_output]) for i_input in input_channels, i_output in output_channels])
    padded = Array{eltype(fftPSFs)}(undef, psfL, psfL, input_channels, output_channels)
    Gop(fftPSFs, objL, imgL, input_channels, output_channels, padded) 
end

Base.size(G::Gop) = (G.output_channels * G.imgL^2, G.input_channels * G.objL^2) 
const GopTranspose = LinearMaps.TransposeMap{<:Any, <:Gop}
    
function Base.:(*)(G::Gop, uflat::AbstractVector)
    u = reshape(uflat, (G.objL, G.objL, G.input_channels))
    
    to_y(obj_plane, kernel) = real.(convolve(obj_plane, kernel))
    utmp = [u[:, :, i_input] for i_input in 1:G.input_channels, i_output in 1:G.output_channels] # TODO: avoid this
    fftPSFstmp = [G.fftPSFs[:, :, i_input, i_output] for i_input in 1:G.input_channels, i_output in 1:G.output_channels] # TODO: store on workers
    y = map(to_y, utmp, fftPSFstmp) 
    y = sum(y, dims=1)
    y = stack(y)

    y[:]
end

function LinearMaps._unsafe_mul!(yflat::AbstractVector, G::Gop, uflat::AbstractVector)
    u = reshape(uflat, (G.objL, G.objL, G.input_channels))
    y = reshape(yflat, (G.imgL, G.imgL, G.output_channels))
    y .= 0

    # TODO: figure out best way of doing this.
    # ideally, in-place, fast, and parallelized over all types of channels
    # (wait this is already in-place, awesome)

	outs = Array{AbstractArray}(undef, (G.input_channels, G.output_channels))
	Threads.@threads for (i_input, i_output) in collect(Iterators.product(1:G.input_channels, 1:G.output_channels))
        @views out = real.(convolve!(u[:, :, i_input], G.fftPSFs[:, :, i_input, i_output], G.padded[:, :, i_input, i_output]))
		outs[i_input, i_output] = out
	end

	Threads.@threads for i_output in 1:G.output_channels
		y[:, :, i_output] = ThreadsX.sum(outs[i_input, i_output] for i_input in 1:G.input_channels)
	end

    yflat
end

function Base.:(*)(Gt::GopTranspose, yflat::AbstractVector)
    G = Gt.lmap
    y = reshape(yflat, (G.imgL, G.imgL, G.output_channels))
    
    to_u(img_conf, kernel) = real.(convolveT(img_conf, kernel))
    ytmp = [y[:, :, i_output] for i_input in 1:G.input_channels, i_output in 1:G.output_channels] # TODO: avoid this
    fftPSFstmp = [G.fftPSFs[:, :, i_input, i_output] for i_input in 1:G.input_channels, i_output in 1:G.output_channels]

    u = map(to_u, ytmp, fftPSFstmp)
    

    u = sum(u, dims=2)
    u = stack(u)
    u[:]
end

function LinearMaps._unsafe_mul!(uflat::AbstractVector, Gt::GopTranspose, yflat::AbstractVector)
    G = Gt.lmap
    y = reshape(yflat, (G.imgL, G.imgL, G.output_channels))
    u = reshape(uflat, (G.objL, G.objL, G.input_channels))
    u .= 0

    Threads.@threads for i_input in 1:G.input_channels
        for i_output in 1:G.output_channels
            @views u[:, :, i_input] .+= real.(convolveT!(y[:, :, i_output], G.fftPSFs[:, :, i_input, i_output], G.padded[:, :, i_input, i_output]))
        end
    end

    uflat
end