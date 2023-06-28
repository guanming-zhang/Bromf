include("./src/utils.jl")
using Plots
using Printf
# this line sets the headless mode for julia plotting
# since there is no graphic interface in the cluster
# seed https://discourse.julialang.org/t/plotting-from-a-server/74345
ENV["GKSwstype"] = "100"

if length(ARGS) < 1 
    error("Please specify the data directory")
end
data_dir = ARGS[1] 
dl = DataLoader(data_dir)
x_max,y_max = dl.info["range"][1],dl.info["range"][2]
delta_x = dl.info["range"][1] / dl.info["npts"][1]
delta_y = dl.info["range"][2] / dl.info["npts"][2]
nx, ny = dl.info["npts"][1],dl.info["npts"][2]
x = range(delta_x, x_max;length = nx)
y = range(delta_y, y_max;length = ny)

anim = @animate while !is_eod(dl)
    n_step,data = load(dl)
    t = @sprintf("%4.3f",n_step*dl.info["dt"])
    rho = reshape(Array{Float64}(data["rho"]),(nx,ny))
    heatmap(x,y,rho',aspect_ratio=:equal,tellheight=true,interpolate=true,title= "t = $t")
    xlims!(0,x_max)
    ylims!(0,y_max)
end
mp4(anim, joinpath(data_dir,"density.mp4"))
