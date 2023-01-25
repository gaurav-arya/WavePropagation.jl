function make_model(f, ϵsub, ϵ, cellL, thickness, order, lb, ub, filename)
    # TODO: call out to Python
    pts = chebpoints(order, lb, ub)
end

function get_model(order, lb, ub, filename)
    f = open(filename)
    function val(line)
        dat = split(line)
        parse(Float64, dat[1]) + parse(Float64, dat[2]) * im
    end
    vals = [val(line) for line in eachline(f)]
    chebinterp(vals, lb, ub)
end
