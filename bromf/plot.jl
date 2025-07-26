include("./src/utils.jl")
using Plots
using Printf
using Statistics
using CSV
using DataFrames

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
N = dl.info["N"]
x = range(delta_x, x_max;length = nx)
y = range(delta_y, y_max;length = ny)
sum_rho = Array{Float64,1}()
var_rho = Array{Float64,1}()
time = Array{Float64,1}()
n_step_list = Array{Float64,1}()


anim = @animate while !is_eod(dl)
    n_step,data = load(dl)
    # t = @sprintf("%4.3f",n_step*dl.info["dt"])
    t = data["t"]
    rho = reshape(Array{Float64}(data["rho"]),(nx,ny))

    heatmap(x,y,rho',aspect_ratio=:equal,tellheight=true,interpolate=true,title= "t = $t")
    xlims!(0,x_max)
    ylims!(0,y_max)

    push!(sum_rho,sum(rho))
    push!(var_rho,var((rho ./ N/(x_max*y_max)) .- 1.0))
    push!(time, t)
    push!(n_step_list, n_step)
end
mp4(anim, joinpath(data_dir,"density.mp4"))

p = scatter(time, sum_rho, marker = 'o', title="sum rho")
savefig(p, joinpath(data_dir,"sum_rho.png"))

p = scatter(time, var_rho, marker = 'o', title="variance rho")
savefig(p, joinpath(data_dir,"var_rho.png"))

# Create a DataFrame
# df_sum_rho = DataFrame(N_Step=n_step_list, Time=time, Sum_Rho=sum_rho)
df_var_rho = DataFrame(N_Step=n_step_list, Time=time, Var_Rho=var_rho)

# Save the DataFrame to a .csv file
# CSV.write(joinpath(data_dir, "sum_rho.csv"), df_sum_rho)
CSV.write(joinpath(data_dir, "var_rho.csv"), df_var_rho)