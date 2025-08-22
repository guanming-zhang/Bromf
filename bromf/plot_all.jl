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
x = range(delta_x, x_max;length = nx)
y = range(delta_y, y_max;length = ny)
N = dl.info["N"]
phi = dl.info["phi"]
# R = sqrt((phi*x_max*y_max)/(pi*N))
R = dl.info["R"]

#lists to store the time series data of different variables
n_step_list = Any[]
t_list = Any[]
# rho_list = Any[]
# kx_list = Any[]
# ky_list = Any[]
# s_2d_list = Any[]
k_list = Any[]
s_list = Any[]
# r_list = Any[]
# corr_list = Any[]


anim = @animate while !is_eod(dl)
    n_step,data = load(dl)
    # t = @sprintf("%4.3f",n_step*dl.info["dt"])
    t = data["t"]
    rho = reshape(Array{Float64}(data["rho"]),(nx,ny))
    # plot density
    p1 = heatmap(x,y,rho',aspect_ratio=:equal,tellheight=true,interpolate=true,title="t = $t")
    #plot normalized density
    # p1 = heatmap(x,y,rho'/(N/(x_max*y_max)),aspect_ratio=:equal,tellheight=true,interpolate=true,title="t = $t")
    xlims!(p1,0,x_max)
    ylims!(p1,0,y_max)

    kx,ky,s_2d = get_structure_factor(rho,nx,ny,x_max,y_max,N)

    # #make the central point to be 1, so that its log is 0
    # s_2d_temp = collect(s_2d)
    # s_2d_temp[div(nx,2)+1, div(ny,2)+1] = 1.0
    # plot the struture factor
    # p2 = heatmap(kx,ky,log10.(s_2d_temp)',aspect_ratio=:equal,tellheight=true,interpolate=true,title="log10 S(kx,ky)")
    p2 = heatmap(kx,ky,log10.(s_2d)',aspect_ratio=:equal,tellheight=true,interpolate=true,title="log10 S(kx,ky)")
    xlims!(p2,kx[1],kx[end])
    ylims!(p2,kx[1],ky[end])
    # xlims!(p2,-1*(2.0*pi)/(2.0*R),(2.0*pi)/(2.0*R))
    # ylims!(p2,-1*(2.0*pi)/(2.0*R),(2.0*pi)/(2.0*R))

    #plot the radial structure factor
    # k,s = get_radial_structure_factor(kx,ky,s_2d)
    k,s = get_radial_profile(s_2d)
    k = Float64.(k) .* (maximum(kx)/maximum(k)*R*2.0/(2.0*pi))
    p3 = scatter(k,s,xscale=:log10, yscale=:log10, legend=false, title="S(k)")

    # k = collect(k)
    # s = collect(s)
    # plot_x_min = -Inf
    # plot_x_max = (2.0*pi)/(2.0*R)
    # indices = (k .>= plot_x_min) .& (k .<= plot_x_max)
    # k_subset = k[indices]
    # s_subset = s[indices]
    # p3 = scatter(k_subset, s_subset, xscale=:log10, legend=false, yscale=:log10, title="S(k)")

    #plot the radial correlation function
    r,corr = get_radial_cross_corr(rho,nx,ny,x_max,y_max)
    p4 = scatter(r,corr,xscale=:log10,legend=false,title="Corr(r)")
    
    p = plot(p1,p2,p3,p4,layout=(2,2))

    #push data to time series list
    push!(n_step_list, n_step)
    push!(t_list, t)
    # push!(rho_list, rho)
    # push!(kx_list, kx)
    # push!(ky_list, ky)
    # push!(s_2d_list, s_2d)
    push!(k_list, k)
    push!(s_list, s)
    # push!(r_list, r)
    # push!(corr_list, corr)

end
mp4(anim, joinpath(data_dir,"all.mp4")) 


#save data as csv files
# Create DataFrames
# df_rho = DataFrame(N_Step=n_step_list, Time=t_list, Rho=rho_list)
# df_s_2d = DataFrame(N_Step=n_step_list, Time=t_list, KX=kx_list, KY=ky_list, S_2D=s_2d_list)
df_s = DataFrame(N_Step=n_step_list, Time=t_list, K=k_list, S=s_list)
# df_corr = DataFrame(N_Step=n_step_list, Time=t_list, R_r=r_list, Corr=corr_list)

# Save the DataFrames to .csv files
# CSV.write(joinpath(data_dir, "rho_time_series.csv"), df_rho)
# CSV.write(joinpath(data_dir, "s_2d_time_series.csv"), df_s_2d)
CSV.write(joinpath(data_dir, "s_time_series.csv"), df_s)
# CSV.write(joinpath(data_dir, "corr_time_series.csv"), df_corr)


