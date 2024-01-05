import Base.joinpath
using SparseArrays
using JSON
using Serialization
using InteractiveUtils
using Base.Threads
using DifferentialEquations

include("finite_diff.jl")


mutable struct NumericalMeanField2D
    rng::Array{Float64,1}         # size of the domain is rng[1] by rng[2] for [0,rng[1]]*[0,rng[2]]
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
    # sparse central difference matrix block_cdiff_mat::Dict{Tuple{Integer,Integer},SparseMatrixCSC{Float64, Int64}}
    block_cdiff_mat::Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}
    # sparse froward difference matrix block_cdiff_mat::Dict{Tuple{Integer,Integer},SparseMatrixCSC{Float64, Int64}}
    block_fdiff_mat::Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}
    # model variables
    rho::Array{Float64,1}         # density, a vector of N*N points (N = rng[1]*rng[2])
    drho::Array{Float64,1}        # change of density at each step
    sigma::Array{Float64,3}       # stress sigma, a 2*2*(N^2) matrix sigma[1,1,:] means sigma_xx
    potential_kernel::Array{Float64,2}  # the kernel for potential functional
    stress_kernel::Dict{Tuple{Int64,Int64},Array{Float64,2}}  # the kernel for stress calculation
    mu::Array{Float64,1}          # chemical potential, a N^2-elements vector
    j::Array{Float64,2}           # j energy flux, a 2*(N^2) matrix j = div
    f::Array{Float64,2}           # f force density, a 2*(N^2) matrix f = div. sigma
    #noise term addition
    white_noise::Array{Float64,1}                                # a Gaussian white noise vector of N*N points (N = rng[1]*rng[2])
    multi_noise::Dict{Tuple{Int64},Array{Float64,2}}             # multiplicative noise =white noise*sqrt(rho), a 2*(N^2) matrix, multi_noise[1,:] means multi_noise_x
    force_noise::Array{Float64,2}                                # noise force, a 2*(N^2) matrix, force_noise[1,:] means force_noise_x
    noise_kernel::Dict{Tuple{Int64,Int64},Array{Float64,2}}      # the kernel for noise calculation

    # auxiliary variables
    rho_store::Array{Float64,1}   # an intermediate variable for predictor-corrector scheme
    drho_store::Array{Float64,1}  # an intermediate varialbe for predictor-corrector scheme
    # flag variables
    is_initialized::Bool          # a flag showing if the initial condition is set
    is_parameters_set::Bool       # a flag showing if the parameters are set 
    # Julia integrator(initilaized as nothing)
    integrator::Any                 
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
    x = range(delta[1], x_max;length = nx)
    y = range(delta[2], y_max;length = ny)
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

    # generate block central difference matrix
    block_cdiff_mat=Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}()
    for odiff_x in 0:2
        for odiff_y in 0:2
            if odiff_x + odiff_y <=2 && odiff_x + odiff_y>0 
                odiff = (odiff_x,odiff_y)
                block_cdiff_mat[odiff] = cut_into_blocks(mixed_diff_mat2d(odiff,nx,ny,delta[1],delta[2],"central",2),num_th)
            end
        end
    end

    # generate block forward difference matrix
    block_fdiff_mat=Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}()
    for odiff_x in 0:2
        for odiff_y in 0:2
            if odiff_x + odiff_y <=2 && odiff_x + odiff_y>0 
                odiff = (odiff_x,odiff_y)
                block_fdiff_mat[odiff] = cut_into_blocks(mixed_diff_mat2d(odiff,nx,ny,delta[1],delta[2],"forward",1),num_th)
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
    # initialize the kernel, they will be reset later
    potential_kernel = zeros(Float64,21,21)
    stress_kernel = Dict{Tuple{Int64,Int64},Array{Float64,2}}()
    #noise_term_additions
    white_noise = zeros(Float64, nx * ny)
    multi_noise = Dict{Tuple{Int64},Array{Float64,2}}()        
    force_noise = zeros(Float64, 2, nx * ny)       
    noise_kernel = Dict{Tuple{Int64,Int64},Array{Float64,2}}() 
    NumericalMeanField2D(rng, npts, delta, time_scheme, params, dt, step_counter, num_th, pts_per_th,
        x, y, block_cdiff_mat, block_fdiff_mat, rho, drho, sigma, potential_kernel, stress_kernel, mu, j, f, white_noise,
        multi_noise, force_noise, noise_kernel, rho_store, drho_store, false, false, nothing)
end


function set_model_params(model::NumericalMeanField2D, T, D, R, Gamma=1.0)
    model.params["T"] = T
    model.params["D"] = D # strength of the stress
    model.params["R"] = R # the radius of the particles
    model.params["Gamma"] = Gamma
    if sqrt(model.rng[1]*model.rng[2]) < 4.0*R
        error("The domain size must be larger than 4R")
    end
    set_kernels(model)
    model.is_parameters_set = true
end

function set_kernels(model::NumericalMeanField2D)
    # the kernel is initialized here
    # kernel is a function k(r)
    # for BRO potential kernel k(r) = R-0.5r (=0 if r>2R)
    # for BRO stress kernel s_xx(r) = 0.25*rx*rx/r^2 (=0 if r>2R)
    #                       s_xy(r) = 0.25*rx*ry/r^2 (=0 if r>2R)
    #                       s_yy(r) = 0.25*ry*ry/r^2 (=0 if r>2R)
    
    Nx = trunc(Int, 2.0*model.R/model.delta[1]) + 4 # 4 is a small number for marginal padding 
    Ny = trunc(Int, 2.0*model.R/model.delta[2]) + 4 # 4 is a small number for marginal padding
    if (Nx > model.npts[0] || Ny > model.npts[0]) 
        error("[convolusion fails] R is too big, decrease R")
    end
    if (model.R < model.delta[1]) 
        error("[convolusion fails] R is too small, finer the grids or increase R")
    end
    for alpha in 1:2
        for beta in 1:2
            model.stress_kernel[(alpha,beta)] = zeros(Float64,2*Nx+1,2*Ny+1)
            model.noise_kernel[(alpha,beta)] = zeros(Float64,2*Nx+1,2*Ny+1)
        end
    end
    model.potential_kernel = zeros(Float64,2*Nx+1,2*Ny+1)
    dx,dy = model.delta[1],model.delta[2]
    for ix in -Nx:Nx
        for iy in -Ny:Ny
            x = (ix + Nx)*dx
            y = (ix + Ny)*dy
            r = sqrt(x*x+ y*y)
            r2 = r*r
            if r <= 2*R
                # potential
                model.potential_kernel[ix,iy] = model.R - 0.5*r
                # stress
                model.stress_kernel[(1,1)] = 0.25*x*x/r2
                model.stress_kernel[(1,2)] = 0.25*x*y/r2
                model.stress_kernel[(2,1)] = 0.25*y*x/r2
                model.stress_kernel[(2,2)] = 0.25*y*y/r2
                
                model.noise_kernel[(1,1)] = 0.5*ix*ix*dx*dx/r
                model.noise_kernel[(1,2)] = 0.5*ix*iy*dx*dy/r
                model.noise_kernel[(2,1)] = 0.5*iy*ix*dx*dy/r
                model.noise_kernel[(2,2)] = 0.5*iy*iy*dy*dy/r
            end
        end
    end
end

function set_initial_condition(model::NumericalMeanField2D, rho::Array{Float64,2})
    model.rho = reshape(rho, model.npts[1] * model.npts[2])
    model.rho_store = reshape(rho, model.npts[1] * model.npts[2])
    model.step_counter = 0
    # we warp the model into julia ODE solver
    # tmax = 10000.0 is a large number to make sure t is in [0,tmax]
    # To avoid out-of-memory error caused by storing a large amount of data 
    # to the computer memory, we set save_on = false to avoid the aforementioned problem 
    # and then save the data to the hard disk at a regular time interval in calculation.jl.
    if model.time_scheme == "julia-Tsit5"
        prob = ODEProblem(wrapped_update!,model.rho,(0.0,1000.0),model)
        model.integrator = init(prob,Tsit5();save_on=false)
    elseif model.time_scheme == "julia-RK4"
        prob = ODEProblem(wrapped_update!,model.rho,(0.0,1000.0),model)
        model.integrator = init(prob,RK4();save_on=false)
    elseif model.time_scheme =="julia-TRBDF2"
        prob = ODEProblem(wrapped_update!,model.rho,(0.0,1000.0),model)
        model.integrator = init(prob,TRBDF2();save_on=false) 
    end
    model.is_initialized = true
end

function update_chemical_pot_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    # evaluate the chemical potential
    rho = reshape(model.rho,(model.npts[1],model.npts[2]))
    for idx in idx_rng
        # the chemical potential at (ix*dx,iy*dy)
        ix = mod_idx(idx,model.npts[1]) 
        iy = div(idx - ix,model.npts[1]) + 1
        model.mu[idx] = model.rho[idx]*convol2d(rho,model.potential_kernel,(ix,iy),model.delta[1],model.delta[2]) 
    end
end

function update_stress_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    # calculate the stress
    rho = reshape(model.rho,(model.npts[1],model.npts[2]))
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    for alpha in 1:2
        for beta in 1:2
            odiff = [0, 0]
            odiff[alpha] += 1
            odiff[beta] += 1
            for idx in idx_rng
                # the stress at (ix*dx,iy*dy)
                ix = mod_idx(idx,model.npts[1]) 
                iy = div(idx - ix,model.npts[1]) + 1
                model.sigma[alpha, beta, idx] =  -0.5*model.params["D"]*model.rho[idx]*convol2d(
                                                 rho,model.stress_kernel[(alpha,beta)],(ix,iy),model.delta[1],model.delta[2])
            end
        end
    end
end

function update_noise_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    # calculate the force in noise term
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    # assign zero to model.force_noise
    # try not to use fill since it will change the entire matrix
    for i in idx_rng
        for alpha in 1:2
            model.force_noise[alpha,i] = 0.0
        end
        #failsafe - floor the negative values of rho to zero
        if model.rho[i] < 0.0
            model.rho[i] = 0.0
        end
    end
    rho = reshape(model.rho,(model.npts[1],model.npts[2]))

    for alpha in 1:2
        # sample a matrix of model.npts[1]*model.npts[2] Gaussian random variables with mean = 0 and sd = sqrt(dt)
        model.white_noise = randn(model.npts[1]*model.npts[2])
        white_noise = reshape(sqrt(model.dt)*model.white_noise, (model.npts[1],model.npts[2]))
        model.multi_noise[(alpha,)] = sqrt.(rho).*white_noise

        for beta in 1:2
            for idx in idx_rng
                # the noise stress at (ix*dx,iy*dy)
                ix = mod_idx(idx,model.npts[1]) 
                iy = div(idx - ix,model.npts[1]) + 1
                model.force_noise[alpha, idx] += sqrt(model.params["D"])*model.rho[idx]*corr2d(
                    model.noise_kernel[(alpha,beta)],model.multi_noise[(alpha,)],(ix,iy),model.delta[1],model.delta[2])
            end
        end
    end
end

function calculate_flux_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    # calculate the free energy flux j
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        grad_mu = model.block_cdiff_mat[tuple(odiff...)][th_num] * model.mu
        grad_rho = model.block_cdiff_mat[tuple(odiff...)][th_num] * model.rho
        model.j[alpha, idx_rng] = (model.rho[idx_rng] .* grad_mu / model.params["Gamma"]
                                + model.params["T"] * grad_rho / model.params["Gamma"])
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
end



function update_drho_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    model.drho[idx_rng] .= 0.0
    
    # use chain rule to break ∇.(rho ∇mu) into ∇rho.∇mu + rho ∇^2 mu 
    # also calculate T(∇^2)rho
    # calculate the ∇rho.∇mu using the forward difference scheme
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        grad_mu = model.block_fdiff_mat[tuple(odiff...)][th_num] * model.mu
        grad_rho = model.block_fdiff_mat[tuple(odiff...)][th_num] * model.rho
        model.drho[idx_rng] += grad_mu.*grad_rho
    end
    
    # calculate (rho (∇^2)mu) + T(∇^2)rho using the central difference scheme
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 2
        # calculate rho (∇^2)mu
        model.drho[idx_rng] += model.rho[idx_rng].*(model.block_cdiff_mat[tuple(odiff...)][th_num] * model.mu)
        # calcuate T(∇^2)rho
        model.drho[idx_rng] += model.params["T"]*model.block_cdiff_mat[tuple(odiff...)][th_num] * model.rho
    end

    # divided by the mobility coeff,Gamma
    model.drho /= model.params["Gamma"]

    # calculate the stress part ∇∇:sigma
    for alpha in 1:2
        for beta in 1:2
            odiff = [0, 0]
            odiff[alpha] += 1
            odiff[beta] += 1
            model.drho[idx_rng] += model.block_cdiff_mat[tuple(odiff...)][th_num] * model.sigma[alpha, beta, :]
        end
    end

    # calculate the noise part ∇.force_noise
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        model.drho[idx_rng] += (model.block_cdiff_mat[tuple(odiff...)][th_num] * model.force_noise[alpha, :])./model.dt
        #change the division by dt
    end

    # #white noise for testing
    # for alpha in 1:2
    #     odiff = [0, 0]
    #     # odiff[alpha] = 1
          #Dean's noise
    #     # model.drho[idx_rng] += (model.block_cdiff_mat[tuple(odiff...)][th_num] * (model.params["D"]*sqrt(model.dt)*(sqrt.(model.rho)).*randn(model.npts[1]*model.npts[2])))./model.dt
          #Div(white noise)
    #     # model.drho[idx_rng] += (model.block_cdiff_mat[tuple(odiff...)][th_num] * (model.params["D"]*sqrt(model.dt).*randn(model.npts[1]*model.npts[2])))./model.dt

    # end
end

function update_parallel!(model::NumericalMeanField2D)
    # note that these parallel bolcks cannot be merged in one thread for loop
    Threads.@threads for th_num in 1:model.num_th
        update_chemical_pot_parallel!(model,model.pts_per_th,th_num)
    end
    Threads.@threads for th_num in 1:model.num_th
        update_stress_parallel!(model,model.pts_per_th,th_num)
    end
    Threads.@threads for th_num in 1:model.num_th
        update_noise_parallel!(model,model.pts_per_th,th_num)
    end
    Threads.@threads for th_num in 1:model.num_th
        update_drho_parallel!(model,model.pts_per_th,th_num)
    end
end

function wrapped_update!(drho,rho,model,t)
    """
    we wrap the update_drho_parallel function to make it compatable to Julia ODE solver
    which is of the form du/dt = f(du,u,p,t) where u is the unknown, p is the parameter
    in our case we update drho/dt = wrapped_update(drhorho,model,t) where we use the model as
    the parameter
    """
    model.rho = rho
    update_parallel!(model)
    drho .= model.drho
end


function one_step(model::NumericalMeanField2D)
    if !model.is_initialized
        error("Please set the initialcondition before fowarding in time")
    elseif !model.is_parameters_set
        error("Please specify the value of model prameters before fowarding in time")
    end
    if model.time_scheme == "predictor-corrector"
        model.rho_store = copy(model.rho)
        update_parallel!(model)
        model.rho += model.drho * model.dt
        model.drho_store = copy(model.drho) # copy by value
        Threads.@threads for th_num in 1:model.num_th
            update_drho_parallel!(model,model.pts_per_th,th_num)
        end
        model.rho = model.rho_store + 0.5 * model.dt * (model.drho + model.drho_store)
    elseif model.time_scheme == "RK2"
        model.rho_store = copy(model.rho)
        # calclate k1 and update rho_new
        update_parallel!(model)
        k1 = model.dt*model.drho
        # calclate k2 and update rho_new
        model.rho = model.rho_store + k1
        update_parallel!(model)
        k2 = model.dt*model.drho
        model.rho = model.rho_store + 0.5*(k1 + k2)
    elseif model.time_scheme == "forward-Euler"
        update_parallel!(model)
        model.rho += model.drho * model.dt
    else
        error("one_step() only works for time-scheme = 
        [predictor-corrector, forward-Euler, RK2]")
    end
    model.step_counter += 1
end 

function n_steps(model::NumericalMeanField2D,n)
    if !model.is_initialized
        error("Please set the initialcondition before fowarding in time")
    elseif !model.is_parameters_set
        error("Please specify the value of model prameters before fowarding in time")
    end

    if model.time_scheme in ["julia-Tsit5","julia-TRBDF2","julia-RK4"]
        step!(model.integrator,n*model.dt,true)
        model.rho = model.integrator.u
    else 
        error("n_steps() only works for time-scheme = 
              [julia-Tsit5, julia-TRBDF2, julia-RK4]")
    end
    model.step_counter += n
end

function save_data(model::NumericalMeanField2D, dir_str::String,compression::Bool)
    pts_per_thread = div(model.npts[1] * model.npts[2],model.num_th)
    if mod(model.npts[1]*model.npts[2],model.num_th) == 0
        Threads.@threads for th_num in 1:model.num_th
            calculate_flux_parallel!(model,pts_per_thread,th_num)
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
# Try not to use indexing on sparse matrices since 
# it is very slow
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
    model.mu = -2.0*model.params["A2"] * model.rho + 3.0*model.params["A3"] * model.rho .^ 2- 2.0*model.params["K"] * laplace_rho
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
