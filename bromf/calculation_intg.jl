# include("./src/mean_field_model_integral_no_malloc.jl")
# include("./src/mean_field_model_integral_no_malloc_new_pot.jl")
# include("./src/mean_field_model_integral_no_malloc_pairwise.jl")
include("./src/mean_field_model_integral_no_malloc_pairwise_cons_COM.jl")

include("./src/utils.jl")
using Random, Distributions
using Printf
if length(ARGS) < 1 
    error("Please specify the data directory containing input.json ")
end
data_dir = ARGS[1] 
input_data = read_input(Base.joinpath(data_dir,"input.json"))
x_max,y_max = input_data["range"]
nx,ny = input_data["npts"]
dx,dy = x_max/nx, y_max/ny
dt = input_data["dt"]
# dt = input_data["rel_epsilon"]*2.0*input_data["R"]
#model parameters
if haskey(input_data,"rel_epsilon") && haskey(input_data,"N")
    S = 2.0*pi
    d = 2
    if haskey(input_data,"R") 
        R = input_data["R"]
        phi = (input_data["N"]*pi*R*R)/(x_max*y_max)
        input_data["phi"] = phi
        println("phi is overwriten and adjusted based on the vaule of R")
    elseif haskey(input_data,"phi")
        R = sqrt(input_data["phi"]*x_max*y_max/(input_data["N"]*pi))
        input_data["R"] = R
        println("R is overwriten and adjusted based on the vaule of phi")
    else
        error("Please specify R or phi")
    end
    eps = input_data["rel_epsilon"]*2.0*R

    #change D and (1/gamma) according to RO, BRO, and SGD
    if input_data["system"] == "RO"
        input_data["Gamma_inv"] = 0.0
        input_data["D"] = eps*eps*(1.0/(3.0*d))
    elseif input_data["system"] == "BRO"
        input_data["Gamma_inv"] = eps*R
        input_data["D"] = eps*eps*(1.0/3.0)    
    elseif input_data["system"] == "SGD"
        input_data["Gamma_inv"] = 1.0*(2*R)*0.5                      #alpha*b_f, alpha = 1*2R, bf = 0.5
        input_data["D"] = 4*(1.0*(2*R)*1.0*(2*R))*0.5*0.5            #4*alpha^2*bf*(1-bf)
    else
        error("Please specify the system type: RO, BRO, or SGD")
    end

    # overwrite the old input file
    file_path = Base.joinpath(data_dir,"input.json")
    open(file_path, "w") do f
        write(f,JSON.json(input_data,4))
    end
end

T = input_data["T"] 
R = input_data["R"]
D = input_data["D"]
Gamma_inv = input_data["Gamma_inv"]     #Gamma_inv = 1/Gamma
corr = input_data["corr"]
println(input_data)

#add this so that FFT is done in parallel
FFTW.set_num_threads(Threads.nthreads())
temp = rand(Complex{Float64},nx,ny)
p = plan_fft(temp, flags=FFTW.MEASURE)
p_i = plan_ifft(temp, flags=FFTW.MEASURE)

model = NumericalMeanField2D(x_max, y_max, nx, ny, dt, R, input_data["time_scheme"], input_data["system"])
set_model_params(model,T,D,R,Gamma_inv,corr)
# set the initial condition
if input_data["iv"] == "Gaussian-profile"
    x = model.x
    y = model.y'
    sx = input_data["iv_sx"]
    sy  = input_data["iv_sy"]
    a = 1.0/(pi*sx*sy)*input_data["N"]
    rho0 = @. a*exp(-((x-0.5*x_max)/sx)^2 -((y-0.5*y_max)/sy)^2)
elseif input_data["iv"] == "random-normal"
    sr = input_data["iv_srho"]
    mu = input_data["N"]/(x_max*y_max)
    rho0 = rand(Normal(mu, sr*mu), nx,ny)
elseif input_data["iv"] == "coarse-grain"
    rho0 = zeros(Float64,nx,ny)
    x = rand(Uniform(0.0,x_max),round(Int, input_data["N"]))
    y = rand(Uniform(0.0,y_max),round(Int, input_data["N"]))
    w = input_data["iv_rel_window"]*R
    for i in 1:round(Int, input_data["N"])
        ix1 = trunc(Int64,(x[i] - w)/dx)
        ix2 = trunc(Int64,(x[i] + w)/dx)
        iy1 = trunc(Int64,(y[i] - w)/dy)
        iy2 = trunc(Int64,(y[i] + w)/dy)
        for ix in ix1:ix2
            for iy in iy1:iy2
                rho0[mod_idx(ix,nx),mod_idx(iy,ny)] += 1.0
            end
        end
    end
    rho0 = rho0/(dx*dy)*input_data["N"]/sum(rho0)
    print(sum(rho0)*dx*dy)
# elseif input_data["iv"] == "dirac-deltas"
#     # rho0 = zeros(Float64,nx,ny)
#     rho0 = fill(1.0, (nx, ny))
#     M = 5000
#     x_coor = rand(1:nx, M)
#     y_coor = rand(1:ny, M)
#     sr = input_data["iv_srho"]
#     mu = (input_data["N"]*nx*ny)/(x_max*y_max*M)
#     for i in 1:M
#         rho0[x_coor[i], y_coor[i]] = rand(Normal(mu, sr*mu))
#     end
end
set_initial_condition(model,rho0)
println("the current step number is $(model.step_counter)")
# save the initial value
save_data(model,data_dir,input_data["compression"]>0)

start_time = time()

for s in 1:input_data["n_steps"]
    if model.time_scheme in ["forward-Euler","predictor-corrector","RK2"] 
        one_step(model, p, p_i)
        # one_step(model)
        #call garbage collector every 1000 steps
        if s%1000==0
            GC.gc()
        end
    end
    if mod(s,input_data["n_save"]) == 0
        if model.time_scheme in ["julia-Tsit5","julia-TRBDF2","julia-RK4"]
            n_steps(model,input_data["n_save"])
        end
        @printf("The current time step number: %i \n", s)
        if any(isnan, model.rho)
            error("NaN detected, program stops")
        end
        save_data(model,data_dir,input_data["compression"]>0)
    end
end

end_time = time()
elapsed_time = end_time - start_time
println("Elapsed time: ", elapsed_time, " seconds")