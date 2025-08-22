import Base.joinpath
using SparseArrays
using JSON
using Serialization
using InteractiveUtils
using Base.Threads
using DifferentialEquations
using LinearAlgebra
using Random
using Future
using Distributions

include("finite_diff.jl")

mutable struct NumericalMeanField2D
    rng::Array{Float64,1}         # size of the domain is rng[1] by rng[2] for [0,rng[1]]*[0,rng[2]]
    npts::Array{Integer,1}        # number of points per dimension npts
    delta::Array{Float64,1}       # spatial discritization 
    time_scheme::String           # forward-euler or predictor-corrector
    system::String                # system type: RO, BRO, or SGD
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
    # white_noise::Array{Float64,1}                              # a Gaussian white noise vector of N*N points (N = rng[1]*rng[2]) 
    multi_noise::Dict{Tuple{Int64},Array{Float64,2}}             # multiplicative noise =white noise*sqrt(rho), a 2*(N^2) matrix, multi_noise[1,:] means multi_noise_x
    force_noise::Array{Float64,2}                                # noise force, a 2*(N^2) matrix, force_noise[1,:] means force_noise_x
    noise_kernel::Dict{Tuple{Int64,Int64},Array{Float64,2}}      # the kernel for noise calculation
    randngs::Array{MersenneTwister, 1}                           # MersenneTwister random number generator
    t::Float64                                                   # time elapsed
    noise_matrix::Array{Float64,5}                               # noise matrix of size alpha*N*N*(2Kx+1)*(2Ky+1), where alpha = 2 for 2D system
    # auxiliary variables
    rho_store::Array{Float64,1}         # an intermediate variable for predictor-corrector scheme
    drho_store::Array{Float64,1}        # an intermediate varialbe for predictor-corrector scheme
    # auxiliary variables for reducing runtime memory allocation 
    # usage: grad_mu_fd[thread_id][lattice_index]
    grad_mu_fd::Array{Float64,2}             # an intermediate variable for gradient of mu using forward difference scheme
    grad_rho_fd::Array{Float64,2}            # an intermediate variable for gradient of rho using forward difference scheme
    grad_mu_cd::Array{Float64,2}             # an intermediate variable for gradient of mu using central difference scheme
    grad_rho_cd::Array{Float64,2}            # an intermediate variable for gradient of rho using central difference scheme
    grad_sigma_cd::Array{Float64,2}          # an intermediate variable for gradient of sigma using central difference scheme
    grad_force_noise_cd::Array{Float64,2}    # an intermediate variable for gradient of force_noise using central difference scheme
    # flag variables
    is_initialized::Bool                # a flag showing if the initial condition is set
    is_parameters_set::Bool             # a flag showing if the parameters are set 
    # Julia integrator(initilaized as nothing)
    integrator::Any       
end

function NumericalMeanField2D(x_max, y_max, nx, ny, dt, R, t_scheme="forward-Euler", sys="RO")
    rng = [x_max, y_max]
    npts = [nx, ny]
    delta = [x_max / nx, y_max / ny]
    time_scheme = t_scheme
    system = sys
    params = Dict{String,Float64}()
    step_counter = 0
    num_th = Threads.nthreads()
    pts_per_th = div(npts[1] * npts[2],num_th)
    t = 0
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
    # use 2nd order accuary central difference and 1st order forward difference to make sure the mass is conserved

    # # generate isotropic central difference matrix
    # block_cdiff_iso_mat=Dict{Tuple{Integer,Integer},Array{SparseMatrixCSC{Float64, Int64},1}}()
    # for odiff_x in 0:2
    #     for odiff_y in 0:2
    #         if odiff_x + odiff_y <=2 && odiff_x + odiff_y>0 
    #             odiff = (odiff_x,odiff_y)
    #             block_cdiff_iso_mat[odiff] = cut_into_blocks(mixed_diff_mat2d(odiff,nx,ny,delta[1],delta[2],"isotropic",2),num_th)
    #         end
    #     end
    # end
    
    rho = zeros(Float64, nx * ny)
    drho = zeros(Float64, nx * ny)
    sigma = zeros(Float64, 2, 2, nx * ny)
    mu = zeros(Float64, nx * ny)
    j = zeros(Float64, 2, nx * ny)
    f = zeros(Float64, 2, nx * ny)
    rho_store = zeros(Float64, nx * ny)
    drho_store = zeros(Float64, nx * ny)

    grad_mu_fd = zeros(Float64, num_th, pts_per_th)
    grad_rho_fd = zeros(Float64, num_th, pts_per_th)
    grad_mu_cd = zeros(Float64, num_th, pts_per_th)
    grad_rho_cd = zeros(Float64, num_th, pts_per_th)
    grad_sigma_cd = zeros(Float64, num_th, pts_per_th)
    grad_force_noise_cd = zeros(Float64, num_th, pts_per_th)

    Kx = trunc(Int, 2.0*R/delta[1]) + 4 # 4 is a small number for marginal padding 
    Ky = trunc(Int, 2.0*R/delta[2]) + 4 # 4 is a small number for marginal padding
    noise_matrix = zeros(Float64, 2, nx, ny, (2*Kx)+1, (2*Ky)+1)
    
    # initialize the kernel, they will be reset later
    potential_kernel = zeros(Float64,21,21)
    stress_kernel = Dict{Tuple{Int64,Int64},Array{Float64,2}}()
    #noise_term_additions
    # white_noise = zeros(Float64, nx * ny)
    multi_noise = Dict{Tuple{Int64},Array{Float64,2}}()        
    force_noise = zeros(Float64, 2, nx * ny)       
    noise_kernel = Dict{Tuple{Int64,Int64},Array{Float64,2}}() 
    
    # Generate an array of num_th RNGs for parallel generation of random numbers
    randngs = Vector{MersenneTwister}(undef, num_th)
    # Initialize the first RNG with a random seed
    randngs[1] = MersenneTwister()
    # Generate the remaining RNGs
    for i in 2:num_th
        # Use randjump to create a new RNG that is 10^20 steps ahead in the sequence to avoid overlap in randn generation
        randngs[i] = Future.randjump(randngs[i-1], big(10)^20)
    end

    NumericalMeanField2D(rng, npts, delta, time_scheme, system, params, dt, step_counter, num_th, pts_per_th,
        x, y, block_cdiff_mat, block_fdiff_mat, rho, drho, sigma, potential_kernel, stress_kernel, mu, j, f,
        multi_noise, force_noise, noise_kernel, randngs, t, noise_matrix, rho_store, drho_store, grad_mu_fd, grad_rho_fd, grad_mu_cd,
        grad_rho_cd, grad_sigma_cd, grad_force_noise_cd, false, false, nothing)
end


function set_model_params(model::NumericalMeanField2D, T, D, R, Gamma_inv, corr)
    model.params["T"] = T
    model.params["D"] = D # strength of the stress
    model.params["R"] = R # the radius of the particles
    model.params["Gamma_inv"] = Gamma_inv
    model.params["corr"] = corr
    if sqrt(model.rng[1]*model.rng[2]) < 4.0*R
        error("The domain size must be larger than 4R")
    end
    set_kernels(model)
    model.is_parameters_set = true
end

function gen_noise_matrix!(model::NumericalMeanField2D)
    # size of the noise matrix = (nx, ny, 2*Kx+1, 2*Ky+1) 

    Kx = trunc(Int, 2.0*model.params["R"]/model.delta[1]) + 4 # 4 is a small number for marginal padding 
    Ky = trunc(Int, 2.0*model.params["R"]/model.delta[2]) + 4 # 4 is a small number for marginal padding
    if (Kx > model.npts[1] || Ky > model.npts[2]) 
        error("[convolusion fails] R is too big, decrease R")
    end
    if (model.params["R"] < model.delta[1]) 
        print("Warning: R is too small, the convolusion vanishes\n")
    end
    
    nx,ny = model.npts[1],model.npts[2]
    dx,dy = model.delta[1],model.delta[2]
    model.noise_matrix = zeros(Float64,2,nx,ny,(2*Kx)+1,(2*Ky)+1)
    # model.noise_matrix = randn(model.randngs[Threads.threadid()],2,nx,ny,(2*Kx)+1,(2*Ky)+1) 

    #center of the matrix
    idx_center_x = Kx + 1
    idx_center_y = Ky + 1

    if (model.params["corr"] == -1.0)

        for i in 1:nx
            for j in 1:ny
                for ix in 1:2*Kx+1
                    for iy in 1:2*Ky+1
                        rel_x = (ix - idx_center_x)
                        rel_y = (iy - idx_center_y)
                        r = sqrt((rel_x*dx)^2 + (rel_y*dy)^2)

                        #conserve center of mass by giving equal and opposite noise
                        for alpha in 1:2
                            if ((r <= 2*model.params["R"]) && (ix != idx_center_x || iy != idx_center_y) && model.noise_matrix[alpha,i,j,ix,iy] == 0.0)
                                # if ix == idx_center_x and iy == idx_center_y then x==y==0
                                model.noise_matrix[alpha,i,j,ix,iy] = randn(model.randngs[Threads.threadid()]) 
                                model.noise_matrix[alpha,mod_idx(i+rel_x,nx),mod_idx(j+rel_y,ny),idx_center_x-rel_x,idx_center_y-rel_y] = -1.0*model.noise_matrix[alpha,i,j,ix,iy]
                            end
                        end
                    end
                end
            end
        end

    else

        #define a 2D gaussian with correlation 'p' to sample correlated noise
        p = model.params["corr"]
        mean_vector = [0.0, 0.0]     #mean vector for the 2D Gaussian
        cov_matrix = [1.0 p; p 1.0]  #covariance matrix
        dist = MvNormal(mean_vector, cov_matrix)

        for i in 1:nx
            for j in 1:ny
                for ix in 1:2*Kx+1
                    for iy in 1:2*Ky+1
                        rel_x = (ix - idx_center_x)
                        rel_y = (iy - idx_center_y)
                        r = sqrt((rel_x*dx)^2 + (rel_y*dy)^2)

                        #correlated noise with correlation 'p'
                        for alpha in 1:2
                            if ((r <= 2*model.params["R"]) && (ix != idx_center_x || iy != idx_center_y) && model.noise_matrix[alpha,i,j,ix,iy] == 0.0)
                                # if ix == idx_center_x and iy == idx_center_y then x==y==0
                                sample = rand(model.randngs[Threads.threadid()], dist)
                                model.noise_matrix[alpha,i,j,ix,iy] = sample[1]
                                model.noise_matrix[alpha,mod_idx(i+rel_x,nx),mod_idx(j+rel_y,ny),idx_center_x-rel_x,idx_center_y-rel_y] = sample[2]
                            end
                        end
                    end
                end
            end
        end

    end

end


function set_kernels(model::NumericalMeanField2D)
    # the kernel is initialized here
    # kernel is a function k(r)
    # for BRO potential kernel k(r) = R-0.5r (=0 if r>2R)

    # for BRO stress kernel s_xx(r) = 0.25*rx*rx/r^2 (=0 if r>2R)
    #                       s_xy(r) = 0.25*rx*ry/r^2 (=0 if r>2R)
    #                       s_yy(r) = 0.25*ry*ry/r^2 (=0 if r>2R)

    # for BRO noise kernel  s_xx(r) = 0.5*rx*rx/r^2 (=0 if r>2R)
    #                       s_xy(r) = 0.5*rx*ry/r^2 (=0 if r>2R)
    #                       s_yy(r) = 0.5*ry*ry/r^2 (=0 if r>2R)
    
    Kx = trunc(Int, 2.0*model.params["R"]/model.delta[1]) + 4 # 4 is a small number for marginal padding 
    Ky = trunc(Int, 2.0*model.params["R"]/model.delta[2]) + 4 # 4 is a small number for marginal padding
    if (Kx > model.npts[1] || Ky > model.npts[2]) 
        error("[convolusion fails] R is too big, decrease R")
    end
    if (model.params["R"] < model.delta[1]) 
        print("Warning: R is too small, the convolusion vanishes\n")
    end
    
    # noise kernel should be defined for real space convolution
    # size of the kernals = (2*Kx+1,2*Ky+1) 
    for alpha in 1:2
        for beta in 1:2                
            model.noise_kernel[(alpha,beta)] = zeros(Float64,2*Kx+1,2*Ky+1)
        end
    end
    # model.potential_kernel = zeros(Float64,2*Kx+1,2*Ky+1)
    dx,dy = model.delta[1],model.delta[2]

    #center of the kernel
    idx_center_x = Kx + 1
    idx_center_y = Ky + 1

    for ix in 1:2*Kx+1
        for iy in 1:2*Ky+1
            x = (ix - idx_center_x)*dx
            y = (iy - idx_center_y)*dy
            r = sqrt(x*x+ y*y)
            r2 = r*r

            if ((r <= 2*model.params["R"]) && (ix != idx_center_x || iy != idx_center_y))
                # if ix == idx_center_x and iy == idx_center_y then x==y==0
                # dividing by r2 or r gives inf, 
                # when r=0 we set everything to 0 to avoid self interaction

                if model.system == "BRO" || model.system == "SGD"
                    #noise kernel BRO and SGD
                    model.noise_kernel[(1,1)][ix,iy] = 0.5*x*x/r2
                    model.noise_kernel[(1,2)][ix,iy] = 0.5*x*y/r2
                    model.noise_kernel[(2,1)][ix,iy] = 0.5*y*x/r2
                    model.noise_kernel[(2,2)][ix,iy] = 0.5*y*y/r2
                elseif model.system == "RO"
                    #noise kernel RO
                    model.noise_kernel[(1,1)][ix,iy] = 1.0
                    model.noise_kernel[(1,2)][ix,iy] = 0.0
                    model.noise_kernel[(2,1)][ix,iy] = 0.0
                    model.noise_kernel[(2,2)][ix,iy] = 1.0

                end
                
            end
        end
    end

    #for convolution using FFT, use this part of code
    #potential and stress kernels are defined for fourier space convolution
    # size of the kernals = (nx,ny) 
    for alpha in 1:2
        for beta in 1:2
            model.stress_kernel[(alpha,beta)] = zeros(Float64,model.npts[1],model.npts[2])
            # model.noise_kernel[(alpha,beta)] = zeros(Float64,model.npts[1],model.npts[2])
        end
        model.multi_noise[(alpha,)] = zeros(Float64,model.npts[1],model.npts[2])
    end
    model.potential_kernel = zeros(Float64,model.npts[1],model.npts[2])
    dx,dy = model.delta[1],model.delta[2]

    #center of the kernel
    idx_center_x = Kx + 1
    idx_center_y = Ky + 1

    #pad the kernel such that the center of the kernel is at position (1,1)
    for ix in 1:2*Kx+1
        for iy in 1:2*Ky+1
            x = (ix - idx_center_x)*dx
            y = (iy - idx_center_y)*dy
            r = sqrt(x*x+ y*y)
            r2 = r*r

            if (ix == idx_center_x && iy == idx_center_y)
                ix_new = mod_idx(ix-Kx,model.npts[1])
                iy_new = mod_idx(iy-Ky,model.npts[2])

                # set potential
                model.potential_kernel[ix_new,iy_new] = model.params["R"] - 0.5*r
            end

            if ((r <= 2*model.params["R"]) && (ix != idx_center_x || iy != idx_center_y))
                # if ix == idx_center_x and iy == idx_center_y then x==y==0
                # dividing by r2 or r gives inf, 
                # when r=0 we set everything to 0 to avoid self interaction

                ix_new = mod_idx(ix-Kx,model.npts[1])
                iy_new = mod_idx(iy-Ky,model.npts[2])

                # set potential
                model.potential_kernel[ix_new,iy_new] = model.params["R"] - 0.5*r

                if model.system == "BRO" || model.system == "SGD"
                    # stress kernel BRO and SGD
                    model.stress_kernel[(1,1)][ix_new,iy_new] = 0.25*x*x/r2
                    model.stress_kernel[(1,2)][ix_new,iy_new] = 0.25*x*y/r2
                    model.stress_kernel[(2,1)][ix_new,iy_new] = 0.25*y*x/r2
                    model.stress_kernel[(2,2)][ix_new,iy_new] = 0.25*y*y/r2
                elseif model.system == "RO"
                    # stress kernel RO 
                    model.stress_kernel[(1,1)][ix_new,iy_new] = 1.0
                    model.stress_kernel[(1,2)][ix_new,iy_new] = 0.0
                    model.stress_kernel[(2,1)][ix_new,iy_new] = 0.0
                    model.stress_kernel[(2,2)][ix_new,iy_new] = 1.0
                end

            end
        end
    end

end

function set_initial_condition(model::NumericalMeanField2D, rho::Array{Float64,2})
    model.rho = reshape(rho, model.npts[1] * model.npts[2])
    model.rho_store = reshape(rho, model.npts[1] * model.npts[2])
    model.step_counter = 0
    model.t = 0
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


function update_chemical_pot_parallel!(model::NumericalMeanField2D,npts_per_th,th_num, p, p_i)
    # evaluate the chemical potential
    rho = reshape(model.rho,(model.npts[1],model.npts[2]))
    model.mu = reshape(convol2d_fft(rho,model.potential_kernel,model.delta[1],model.delta[2], p, p_i), model.npts[1]*model.npts[2]) 
end


function update_stress_parallel!(model::NumericalMeanField2D,npts_per_th,th_num, p, p_i)
    # calculate the stress
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    rho = reshape(model.rho,(model.npts[1],model.npts[2]))
    for alpha in 1:2
        for beta in 1:2
            model.sigma[alpha, beta, :] = reshape(-0.5.*model.params["D"].*rho.*convol2d_fft(
                                                rho,model.stress_kernel[(alpha,beta)],model.delta[1],model.delta[2], p, p_i), model.npts[1]*model.npts[2])              
        end
    end
end


function update_noise_parallel!(model::NumericalMeanField2D,npts_per_th,th_num, p, p_i)
    # calculate the noise term
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    rho = reshape(model.rho,(model.npts[1],model.npts[2]))
    # try not to use fill since it will change the entire matrix
    for alpha in 1:2
        for idx in idx_rng
            model.force_noise[alpha,idx] = 0.0
        end
    end

    for alpha in 1:2
        for beta in 1:2
            #size of noise kernel
            rows, cols = size(model.noise_kernel[(alpha,beta)])
            model.multi_noise[(beta,)] = sqrt.(rho)

            for idx in idx_rng
                # the noise term at (ix*dx,iy*dy)
                ix = mod_idx(idx,model.npts[1]) 
                iy = div(idx - ix,model.npts[1]) + 1

                #pairwise conserved noise
                model.force_noise[alpha, idx] += sqrt(model.params["D"])*sqrt.(model.rho[idx])*convol2d(
                                                    model.multi_noise[(beta,)],model.noise_kernel[(alpha,beta)].*model.noise_matrix[beta,ix,iy,:,:]
                                                    ,(ix,iy),model.delta[1],model.delta[2])
            end
        end
    end


    # #Dean's noise 
    # for alpha in 1:2
    #     for idx in idx_rng
    #         ix = mod_idx(idx,model.npts[1]) 
    #         iy = div(idx - ix,model.npts[1]) + 1
    #         model.force_noise[alpha, idx] += sqrt(model.params["D"])*sqrt(rho[ix,iy])*randn(model.randngs[Threads.threadid()])
    #         # #without sqrt(D) prefactor
    #         # model.force_noise[alpha, idx] += sqrt(rho[ix,iy])*randn(model.randngs[Threads.threadid()])
    #         #model B noise
    #         # model.force_noise[alpha, idx] += sqrt(2*model.params["T"])*randn(model.randngs[Threads.threadid()])
    #     end
    # end

end


function calculate_flux_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    # calculate the free energy flux j
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        @views mul!(model.grad_mu_cd[th_num,:], model.block_cdiff_mat[tuple(odiff...)][th_num], model.mu)
        @views mul!(model.grad_rho_cd[th_num,:], model.block_cdiff_mat[tuple(odiff...)][th_num], model.rho)
        @views @. model.j[alpha, idx_rng] = (model.rho[idx_rng] .* model.grad_mu_cd[th_num,:]
                                            + model.params["T"] .* model.grad_rho_cd[th_num,:]) .* model.params["Gamma_inv"]
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
            @views mul!(model.grad_sigma_cd[th_num,:], model.block_cdiff_mat[tuple(odiff...)][th_num], model.sigma[alpha, beta, :])
            @views @. model.f[alpha,idx_rng] += model.grad_sigma_cd[th_num,:]
        end
    end
end


function update_drho_parallel!(model::NumericalMeanField2D,npts_per_th,th_num)
    idx_rng = npts_per_th*(th_num-1)+1:npts_per_th*th_num
    @views model.drho[idx_rng] .= 0.0
    
    # use chain rule to break ∇.(rho ∇mu) into ∇rho.∇mu + rho ∇^2 mu 
    # also calculate T(∇^2)rho
    # calculate the ∇rho.∇mu using the forward difference scheme
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        @views mul!(model.grad_mu_fd[th_num,:], model.block_fdiff_mat[tuple(odiff...)][th_num], model.mu)
        @views mul!(model.grad_rho_fd[th_num,:], model.block_fdiff_mat[tuple(odiff...)][th_num], model.rho)
        @views @. model.drho[idx_rng] += model.grad_mu_fd[th_num,:] .* model.grad_rho_fd[th_num,:]
    end
    
    # calculate (rho (∇^2)mu) + T(∇^2)rho using the central difference scheme
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 2
        # calculate rho (∇^2)mu
        @views mul!(model.grad_mu_cd[th_num,:], model.block_cdiff_mat[tuple(odiff...)][th_num], model.mu)
        @views @. model.drho[idx_rng] += model.rho[idx_rng] .* model.grad_mu_cd[th_num,:]

        # calcuate T(∇^2)rho
        @views mul!(model.grad_rho_cd[th_num,:], model.block_cdiff_mat[tuple(odiff...)][th_num], model.rho)
        @views @. model.drho[idx_rng] += model.params["T"] * model.grad_rho_cd[th_num,:]
    end

    # divided by the mobility coeff,Gamma, equivalent to multiplying by Gamma_inv = 1/Gamma
    model.drho[idx_rng] .*= model.params["Gamma_inv"] 

    # calculate the stress part ∇∇:sigma
    for alpha in 1:2
        for beta in 1:2
            odiff = [0, 0]
            odiff[alpha] += 1
            odiff[beta] += 1
            @views mul!(model.grad_sigma_cd[th_num,:], model.block_cdiff_mat[tuple(odiff...)][th_num], model.sigma[alpha, beta, :])
            @views @. model.drho[idx_rng] -= model.grad_sigma_cd[th_num,:]
        end
    end

    # calculate the noise part ∇.force_noise
    for alpha in 1:2
        odiff = [0, 0]
        odiff[alpha] = 1
        @views mul!(model.grad_force_noise_cd[th_num,:], model.block_cdiff_mat[tuple(odiff...)][th_num], model.force_noise[alpha, :])
        @views @. model.drho[idx_rng] += model.grad_force_noise_cd[th_num,:] ./ sqrt(model.dt)
    end

end

function update_parallel!(model::NumericalMeanField2D, p, p_i)
    # note that these parallel bolcks cannot be merged in one thread for loop

    #chemical potential and stress are already calculated using FFT which is parallel
    update_chemical_pot_parallel!(model, model.npts[1] * model.npts[2], 1, p, p_i)
    update_stress_parallel!(model, model.npts[1] * model.npts[2], 1, p, p_i)
    
    #generate correlated noise matrix before calculating the noise term, do not do this in parallel
    gen_noise_matrix!(model)

    Threads.@threads for th_num in 1:model.num_th
       update_noise_parallel!(model,model.pts_per_th,th_num, p, p_i)
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


function one_step(model::NumericalMeanField2D, p, p_i)
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
        update_parallel!(model, p, p_i)
        @. model.rho += model.drho .* model.dt

        #increase elapsed time by dt
        model.t += model.dt

        #if rho becomes negative, change dt to dt/1.1
        is_negtive = false
        for i in 1:model.npts[1]*model.npts[2]
            if model.rho[i] < 0.0
                is_negtive = true
                break
            end
        end
        if is_negtive > 0
            @. model.rho -= model.drho .* model.dt
            model.t -= model.dt
            model.dt = model.dt/1.1
        end

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
    # if mod(model.npts[1]*model.npts[2],model.num_th) == 0
    #     Threads.@threads for th_num in 1:model.num_th
    #         calculate_flux_parallel!(model,pts_per_thread,th_num)
    #     end
    # else
    #     error("the number of points must be divisible by the number of thread")
    # end
    file_str = "Frame_$(model.step_counter).json"
    file_path = joinpath(dir_str, file_str)
    # dict_data = Dict("rho" => model.rho, "j" => model.j, "f" => model.f,
    #                  "t"=>model.dt*model.step_counter,"step_num"=>model.step_counter)
    dict_data = Dict("rho" => model.rho,
                     "t"=>model.t,"step_num"=>model.step_counter)
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
        model.j[alpha, :] = model.rho .* grad_mu * model.params["Gamma_inv"]+model.params["T"] * grad_rho * model.params["Gamma_inv"]
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
