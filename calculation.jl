include("./src/mean_field_model.jl")
include("./src/utils.jl")
using Random, Distributions
using Printf
#input_data = read_input(ARGS[1])
data_dir = "../mean_field_model/data"
input_data = read_input(Base.joinpath(data_dir,"input.json"))
x_max,y_max = input_data["range"]
nx,ny = input_data["npts"]
dt = input_data["dt"]
#model parameters
T = input_data["T"]
A = input_data["A"]
B = input_data["B"]
C = input_data["C"]
K = input_data["K"]
Gamma = input_data["Gamma"]
println(input_data)

model = NumericalMeanField2D(x_max, y_max, nx, ny, dt,input_data["time_scheme"])
set_model_params(model,T,A,B,C,Gamma)
# set the initial condition
if input_data["iv"] == "Gaussian-profile"
    x = model.x
    y = model.y'
    sx = input_data["iv_sx"]
    sy = input_data["iv_sy"]
    a = 1.0/(pi*sx*sy)
    rho0 = @. exp(-((x-0.5*x_max)/sx)^2 -((y-0.5*y_max)/sy)^2)
elseif input_data["iv"] == "random-Normal"
    sr = input_data["iv_srho"]
    mu = 1.0/(x_max*y_max)
    rho0 = rand(Normal(mu, sr), nx,ny)
end
set_initial_condition(model,rho0)
println("$(model.step_counter)")
save_data(model,data_dir,true)

for s in range(0,input_data["n_steps"])
    if mod(s,input_data["n_save"]) == 0
        @printf("The current time steps number: %i \n", s)
        if any(isnan, model.rho)
            error("NaN detected, program stops")
        end
        save_data(model,data_dir,input_data["compression"]>0)
    end
    forward(model)
end

println("end of file")