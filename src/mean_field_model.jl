import Base.joinpath
using SparseArrays
using JSON
using Serialization
using InteractiveUtils
using Base.Threads

include("finite_diff.jl")


mutable struct NumericalMeanField2D
    rng::Array{Float64,1}         # size of the domain is rng[1] by rng[2]
    npts::Array{Integer,1}        # number of points per dimension npts
    delta::Array{Float64,1}       # spatial discritization 
    time_scheme::String           # forward-euler or predictor-corrector
    params::Dict{String,Float64}  # model parameters
    dt::Float64                   # time step
    step_counter::Integer         # counter for the current steps
    num_th::Integer               # number of threads
    pts_per_th::Integer           # number of points handled by each thread
    # model variables
    x::Array{Float64,1}           # x coordinate 
    y::Array{Float64,1}           # y coordinate
    #cdiff_mat::Dict{Tuple{Integer,Integer},SparseMatrixCSC{Float64, Int64}} #sparse Totpliz matix for central difference
    block_cdiff_mat::Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}
    # model variables
    rho::Array{Float64,1}         # density, a vector of N*N points (N = rng[1]*rng[2])
    drho::Array{Float64,1}        # change of density at each step
    sigma::Array{Float64,3}       # stress sigma, a 2*2*(N^2) matrix sigma[1,1,:] means sigma_xx
    mu::Array{Float64,1}          # chemical potential, a N^2-elements vector
    j::Array{Float64,2}           # j energy flux, a 2*(N^2) matrix j = div
    f::Array{Float64,2}           # f force density, a 2*(N^2) matrix f = div. sigma
    # auxiliary variables
    rho_store::Array{Float64,1}   # an intermediate variable for predictor-corrector scheme
    drho_store::Array{Float64,1}  # an intermediate varialbe for predictor-corrector scheme
    # flag variables
    is_initialized::Bool          # a flag showing if the initial condition is set
    is_parameters_set::Bool       # a flag showing if the parameters are set 
end

function NumericalMeanField2D(x_max, y_max, nx, ny, dt, t_scheme="forward-Euler")
    rng = [x_max, y_max]
    npts = [nx, ny]
    delta = [x_max / nx, y_max / ny]
    time_scheme = t_scheme
    params = Dict{String,Float64}()
    step_counter = 0
    num_th = Threads.nthreads()
    pts_per_th = div(npts[1] * npts[2],num_th)
    if mod(npts[1]*npts[2],num_th) != 0
        error("the number of points must be divisible by the number of thread")
    end
    x = range(delta[1], x_max, nx)
    y = range(delta[2], y_max, ny)
    #=
    Dx = diff_mat2D(nx, ny, 1, 1) / delta[1]
    Dy = diff_mat2D(nx, ny, 2, 1) / delta[2]
    Dxy = Dx * Dy
    Dxx = diff_mat2D(nx, ny, 1, 2) / (delta[1]^2)
    Dyy = diff_mat2D(nx, ny, 2, 2) / (delta[2]^2)
    cdiff_mat = Dict((1, 0) => Dx, (0, 1) => Dy, (2, 0) => Dxx, (0, 2) => Dyy, (1, 1) => Dxy)
    =#
    if mod(nx*ny,num_th) != 0
        error("the number of points must be divisible by the number of thread")
    end
    block_cdiff_mat=Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}()
    for odiff_x in 0:2
        for odiff_y in 0:2
            if odiff_x + odiff_y <=2 && odiff_x + odiff_y>0 
                odiff = (odiff_x,odiff_y)
                block_cdiff_mat[odiff] = cut_into_blocks(mixed_diff_mat2d(odiff,nx,ny,delta[1],delta[2]),num_th)
            end
        end
    end
    rho = zeros(Float64, nx * ny)
    drho = zeros(Float64, nx * ny)
    sigma = zeros(Float64, 2, 2, nx * ny)
    mu = zeros(Float64, nx * ny)
    j = zeros(Float64, 2, nx * ny)
    f = zeros(Float64, 2, nx * ny)
    rho_store = zeros(Float64, nx * ny)
    drho_store = zeros(Float64, nx * ny)

    NumericalMeanField2D(rng, npts, delta, time_scheme, params, dt, step_counter, num_th, pts_per_th,
        x, y,block_cdiff_mat, rho, drho, sigma, mu, j, f, rho_store, drho_store,false,false)
end


function set_model_params(model::NumericalMeanField2D, T, A, B, C, K, Gamma=1.0)
    model.params["T"] = T
    model.params["A"] = A
    model.params["B"] = B
    model.params["C"] = C
    model.params["K"] = K
    model.params["Gamma"] = Gamma
    model.is_parameters_set = true
end

function set_initial_condition(model::NumericalMeanField2D, rho::Array{Float64,2})
    model.rho = reshape(rho, model.npts[1] * model.npts[2])
    model.rho_store = reshape(rho, model.npts[1] * model.npts[2])
    model.step_counter = 0
    model.is_initialized = true
end



function update_drho_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    laplace_rho = zeros(Float64, npts_per_th)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    
    for i in 1:2
        odiff = [0, 0]
        odiff[i] = 2
        laplace_rho += model.block_cdiff_mat[tuple(odiff...)][th_num] * model.rho
    end
    # mu is the chemical potential due to the interacting free energy
    model.mu[idx_rng] = (-2.0*model.params["A"] * model.rho[idx_rng] 
                        + 3.0*model.params["B"] * model.rho[idx_rng] .^ 2
                        - 2.0*model.params["K"] * laplace_rho)
    
    # calculate the free energy flux j
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        grad_mu = model.block_cdiff_mat[tuple(odiff...)][th_num] * model.mu
        grad_rho = model.block_cdiff_mat[tuple(odiff...)][th_num] * model.rho
        model.j[alpha, idx_rng] = (model.rho[idx_rng] .* grad_mu / model.params["Gamma"]
                                +model.params["T"] * grad_rho / model.params["Gamma"])
    end
    # calculate the stress
    for alpha in 1:2
        for beta in 1:2
            odiff = [0, 0]
            odiff[alpha] += 1
            odiff[beta] += 1
            # we should try to avoid boradcasting(e.g. .* ) when using sparse matrices
            # it will be out of memory error !!!
            model.sigma[alpha, beta, idx_rng] =  model.block_cdiff_mat[tuple(odiff...)][th_num]*model.rho
            model.sigma[alpha, beta, idx_rng] .*=  model.params["C"]*model.rho[idx_rng]
        end
    end
    
    # calculate the force density
    # try not to use fill since it will change the entire matrix
    for i in idx_rng
        for alpha in 1:2
            model.f[alpha,i] = 0.0
        end
    end

    for alpha in 1:2
        for beta in 1:2
            odiff = [0, 0]
            odiff[beta] = 1
            model.f[alpha,idx_rng] += model.block_cdiff_mat[tuple(odiff...)][th_num] * model.sigma[alpha, beta, :]
        end
    end
    for i in idx_rng
        model.drho[i] = 0.0
    end
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        model.drho[idx_rng] += model.block_cdiff_mat[tuple(odiff...)][th_num] * (model.j[alpha, :] - model.f[alpha, :])
    end
end


function forward(model::NumericalMeanField2D)
    if !model.is_initialized
        error("Please set the initialcondition before fowarding in time")
    elseif !model.is_parameters_set
        error("Please specify the value of model prameters before fowarding in time")
    end
    if model.time_scheme == "predictor-corrector"
        model.rho_store = copy(model.rho)
        Threads.@threads for th_num in 1:model.num_th
            update_drho_parallel!(model,model.pts_per_th,th_num)
        end
        model.rho += model.drho * model.dt
        model.drho_store = copy(model.drho) # copy by value
        Threads.@threads for th_num in 1:model.num_th
            update_drho_parallel!(model,model.pts_per_th,th_num)
        end
        model.rho = model.rho_store + 0.5 * model.dt * (model.drho + model.drho_store)
    elseif model.time_scheme == "forward-Euler"
        Threads.@threads for th_num in 1:model.num_th
            update_drho_parallel!(model,model.pts_per_th,th_num)
        end
        model.rho += model.drho * model.dt
    else
        error("time-scheme not defined")
    end
    model.step_counter += 1
end

function save_data(model::NumericalMeanField2D, dir_str::String,compression::Bool)
    pts_per_thread = div(model.npts[1] * model.npts[2],model.num_th)
    if mod(model.npts[1]*model.npts[2],model.num_th) == 0
        Threads.@threads for th_num in 1:model.num_th
            update_drho_parallel!(model,pts_per_thread,th_num)
        end
    else
        error("the number of points must be divisible by the number of thread")
    end
    file_str = "Frame_$(model.step_counter).json"
    file_path = joinpath(dir_str, file_str)
    dict_data = Dict("rho" => model.rho, "j" => model.j, "f" => model.f,
                     "t"=>model.dt*model.step_counter,"step_num"=>model.step_counter)
    json_data = JSON.json(dict_data)
    open(file_path, "w") do f
        write(f, json_data)
    end
    if compression
        zip_file = joinpath(dir_str, "Frame_$(model.step_counter).zip")
        zip_cmd = ["zip","-m", "-j",zip_file,file_path]
        run(`$zip_cmd`)
    end
end


#############--------some notes--------############
# Try not to use boradcasting(.*) on sparse matrices
# or you will get an out-of-memory error in julia
#
###################################################

# unparalleled update function 
#=
function update_drho!(model::NumericalMeanField2D)
    laplace_rho = zeros(Float64, model.npts[1] * model.npts[2])
    for i in 1:2
        odiff = [0, 0]
        odiff[i] = 2
        laplace_rho += model.cdiff_mat[tuple(odiff...)] * model.rho
    end
    # mu is the chemical potential due to the interacting free energy
    model.mu = -2.0*model.params["A"] * model.rho + 3.0*model.params["B"] * model.rho .^ 2- 2.0*model.params["K"] * laplace_rho
    # calculate the free energy flux j
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        grad_mu = model.cdiff_mat[tuple(odiff...)] * model.mu
        grad_rho = model.cdiff_mat[tuple(odiff...)] * model.rho
        model.j[alpha, :] = model.rho .* grad_mu / model.params["Gamma"]+model.params["T"] * grad_rho / model.params["Gamma"]
    end
    # calculate the stress
    for alpha in 1:2
        for beta in 1:2
            odiff = [0, 0]
            odiff[alpha] += 1
            odiff[beta] += 1
            # we should try to avoid boradcasting(e.g. .* ) when using sparse matrices
            # it will be out of memory error !!!
            model.sigma[alpha, beta, :] =  model.cdiff_mat[tuple(odiff...)]*model.rho
            model.sigma[alpha, beta, :] .*=  model.params["C"]*model.rho
        end
    end
    
    # calculate the force density
    fill!(model.f, 0.0)
    for alpha in 1:2
        for beta in 1:2 
            odiff = [0, 0]
            odiff[beta] = 1
            model.f[alpha,:] += model.cdiff_mat[tuple(odiff...)] * model.sigma[alpha, beta, :]
        end
    end
    fill!(model.drho, 0.0)
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        model.drho += model.cdiff_mat[tuple(odiff...)] * (model.j[alpha, :] - model.f[alpha, :])
    end
end
=#



####-----This section is for testing ------#####
#=
using Printf

x_max = y_max = 2.0*pi
nx = ny = 256
N = nx*ny
dt = 0.01
T = 0.1
A = 0.1
B = 0.01
C = 0.01
K = 0.01
Gamma = 1.0

model = NumericalMeanField2D(x_max, y_max, nx, ny, dt)


set_model_params(model, T, A, B, C, K)
x = model.x
y = model.y'
rho = @. sin(x) * cos(2.0 * y)
rho_dx = @. cos(x) * cos(2.0 * y)
rho_dy = @. -2.0 * sin(x) * sin(2.0 * y)

rho_dxy = @. -2.0 * cos(x) * sin(2.0 * y)
rho_dxx = @. -sin(x) * cos(2.0 * y)
rho_dyy = @. -4.0 * sin(x) * cos(2.0 * y)
mu = @. -2.0*A * rho + 3.0*B * rho^ 2 - 2.0*K * (-5.0 * sin(x) * cos(2.0 * y))

j = zeros(Float64, 2, nx, ny)
j[1, :, :] = @. rho * (-2.0*A * rho_dx + 6.0 * B * rho * rho_dx - 2.0*K * (-5.0 * cos(x) * cos(2.0 * y))) + T*rho_dx
j[2, :, :] = @. rho * (-2.0*A * rho_dy + 6.0 * B * rho * rho_dy - 2.0*K * (10.0 * sin(x) * sin(2.0 * y))) + T*rho_dy
f = zeros(Float64, 2, nx, ny)

#sigma_xx = @. -C*(sin(x) * cos(2.0 * y))^2
#sigma_xy = @. -C*0.5 * sin(2.0 * x) * sin(4.0 * yprintln(rho))
#sigma_yy = @. -C*4.0 * (sin(x) * cos(2.0 * y))^2

f = zeros(Float64, 2, nx, ny)
f[1, :, :] = @. C*(-sin(2.0 * x) * (cos(2.0 * y)^2) - 2.0 * sin(2.0 * x) * cos(4.0 * y))
f[2, :, :] = @. C*(-cos(2.0 * x)* sin(4.0 * y) + 8.0 * sin(x)^2 * sin(4.0 * y))

set_initial_condition(model,rho)
@time begin
for i in 1:4000
pts_per_thread = div(model.npts[1] * model.npts[2],model.num_th)
if mod(model.npts[1]*model.npts[2],model.num_th) == 0
    Threads.@threads for th_num in 1:model.num_th
        #update_drho_parallel!(model,pts_per_thread,th_num)
        model.rho[1:div(256*256,1)] 
    end
else
    error("the number of points must be divisible by the number of thread")
end
end
end

rho_fd = reshape(model.rho,nx,ny)
rmse = sqrt(1.0/N*sum((rho_fd - rho).^2))
@printf("RMSE for rho is %1.8f \n",rmse)

# rmse of mu
mu_fd = reshape(model.mu,nx,ny)
rmse_mu = sqrt(1.0/N*sum((mu_fd - mu).^2))
@printf("RMSE for mu is %1.8f \n",rmse_mu)

# rmse of j
for alpha in 1:2
    j_fd = reshape(model.j[alpha, :, :],(nx,ny))
    rmse_j = sqrt(1.0/N*sum((j_fd - j[alpha, :, :]).^2))
    @printf("RMSE for the %i th component of flux, j, is %1.8f \n",alpha,rmse_j)
end

# rmse of f
for alpha in 1:2
    f_fd = reshape(model.f[alpha, :, :],(nx,ny))
    rmse_f = sqrt(1.0/N*sum((f_fd - f[alpha, :, :]).^2))
    @printf("RMSE for the %i th component of force density, f, is %1.8f \n",alpha,rmse_f)
end
=#

