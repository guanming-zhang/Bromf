include("./src/mean_field_model_integral.jl")
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
#model parameters
if haskey(input_data,"rel_epsilon") && haskey(input_data,"N")
    S = 2.0*pi
    d = 2
    if haskey(input_data,"phi")
        R = sqrt(input_data["phi"]*x_max*y_max/(input_data["N"]*pi))
        input_data["R"] = R
        println("R is overwriten and adjusted based on the vaule of phi")
    elseif haskey(input_data,"R") 
        R = input_data["R"]
    else
        error("Please specify R or phi")
    end
    eps = input_data["rel_epsilon"]*2.0*R
    input_data["D"] = eps/12.0
    input_data["Gamma"] = 1.0/eps
    # overwrite the old input file
    file_path = Base.joinpath(data_dir,"input.json")
    open(file_path, "w") do f
        write(f,JSON.json(input_data,4))
    end
end

T = input_data["T"] 
R = input_data["R"]
D = input_data["D"]
Gamma = input_data["Gamma"]
println(input_data)

model = NumericalMeanField2D(x_max, y_max, nx, ny, dt,input_data["time_scheme"])
set_model_params(model,T,D,R,Gamma)
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
    x = rand(Uniform(0.0,x_max),input_data["N"])
    y = rand(Uniform(0.0,y_max),input_data["N"])
    w = input_data["iv_rel_window"]*R
    for i in 1:input_data["N"]
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
end
set_initial_condition(model,rho0)
println("the current step number is $(model.step_counter)")
# save the initial value
save_data(model,data_dir,input_data["compression"]>0)

for s in 1:input_data["n_steps"]
    if model.time_scheme in ["forward-Euler","predictor-corrector","RK2"] 
        one_step(model)
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