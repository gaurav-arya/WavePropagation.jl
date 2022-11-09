using LinearMaps

struct Gop <: LinearMap{Float64}
    fftPSFs::AbstractArray # todo: fix
    objL::Int
    imgL::Int
    input_channels::Int 
    output_channels::Int
    padded::AbstractArray
end   

# TODO: as usual, type better
function Gop(fftPSFs, objL, imgL, input_channels, output_channels)
    psfL = objL + imgL
    padded = Array{ComplexF64}(undef, psfL, psfL, input_channels, output_channels)
    Gop(fftPSFs, objL, imgL, input_channels, output_channels, padded) 
end

Base.size(G::Gop) = (G.output_channels * G.imgL^2, G.input_channels * G.objL^2) 
GopTranspose = LinearMaps.TransposeMap{<:Any, <:Gop} # TODO: make constant
    
function Base.:(*)(G::Gop, uflat::Vector{Float64})
    u = reshape(uflat, (G.objL, G.objL, G.input_channels))
    
    to_y(obj_plane, kernel) = real.(convolve(obj_plane, kernel))
    # uncomment for non-reverse
    # utmp = [u[:, :, i_input, i_output] for i_input in 1:G.input_channels, i_output in 1:G.output_channels] # TODO: avoid this
    utmp = [reverse(u[:, :, i_input, i_output]) for i_input in 1:G.input_channels, i_output in 1:G.output_channels] # TODO: avoid this
    fftPSFstmp = [G.fftPSFs[:, :, i_input, i_output] for i_input in 1:G.input_channels, i_output in 1:G.output_channels] # TODO: store on workers
    y = map(to_y, utmp, fftPSFstmp) 
    y = sum(y, dims=1)
    y = arrarr_to_multi(y)

    y[:]
end
# fake comment
using ThreadsX
#using LoopVectorization

function LinearMaps._unsafe_mul!(yflat::Vector{Float64}, G::Gop, uflat::Vector{Float64})
    u = reshape(uflat, (G.objL, G.objL, G.input_channels))
    y = reshape(yflat, (G.imgL, G.imgL, G.output_channels))
    y .= 0

    # TODO: figure out best way of doing this.
    # ideally, in-place, fast, and parallelized over all types of channels
    # (wait this is already in-place, awesome)

	outs = Array{AbstractArray}(undef, (G.input_channels, G.output_channels))
	Threads.@threads for (i_input, i_output) in collect(Iterators.product(1:G.input_channels, 1:G.output_channels))
		# uncomment for non-reverse
        # @views out = real.(convolve!(u[:, :, i_input], G.fftPSFs[:, :, i_input, i_output], G.padded[:, :, i_input, i_output]))
        @views out = real.(convolve!(reverse(u[:, :, i_input]), G.fftPSFs[:, :, i_input, i_output], G.padded[:, :, i_input, i_output]))
		outs[i_input, i_output] = out
	end

	Threads.@threads for i_output in 1:G.output_channels
		y[:, :, i_output] = ThreadsX.sum(outs[i_input, i_output] for i_input in 1:G.input_channels)
	end

    yflat
end

function Base.:(*)(Gt::GopTranspose, yflat::Vector{Float64})
    G = Gt.lmap
    y = reshape(yflat, (G.imgL, G.imgL, G.output_channels))
    
    to_u(img_conf, kernel) = real.(convolveT(img_conf, kernel))
    ytmp = [y[:, :, i_output] for i_input in 1:G.input_channels, i_output in 1:G.output_channels] # TODO: avoid this
    fftPSFstmp = [G.fftPSFs[:, :, i_input, i_output] for i_input in 1:G.input_channels, i_output in 1:G.output_channels]

    u = map(to_u, ytmp, fftPSFstmp)
    

    u = sum(u, dims=2)
    # uncomment for non-reverse
    # u = arrarr_to_multi(u)
    u = reverse(arrarr_to_multi(u))
    u[:]
end

function LinearMaps._unsafe_mul!(uflat::Vector{Float64}, Gt::GopTranspose, yflat::Vector{Float64})
    G = Gt.lmap
    y = reshape(yflat, (G.imgL, G.imgL, G.output_channels))
    u = reshape(uflat, (G.objL, G.objL, G.input_channels))
    u .= 0

    Threads.@threads for i_input in 1:G.input_channels
        for i_output in 1:G.output_channels
            @views u[:, :, i_input] .+= real.(convolveT!(y[:, :, i_output], G.fftPSFs[:, :, i_input, i_output], G.padded[:, :, i_input, i_output]))
        end
    end

    # uncomment for non-reverse
    # return uflat
    uflat = reverse(uflat)
    uflat
end


# Convert array of arrays to multidimensional array
# using Zygote-friendly operations
function arrarr_to_multi(arrarr)
    outsz = size(arrarr)
    insz = size(arrarr[1])
    arrarr = [reshape(inarr, (prod(insz),)) for inarr in arrarr]
    reshape(vcat(arrarr...), insz..., outsz...)
end