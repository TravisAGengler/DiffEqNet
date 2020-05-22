module drosophila_regulation

using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots

tstart=0.0
tend=10.0
sampling=0.1
data=0 # Data will be set in generate_data

# u0_params in this file refers to the concatination of the u0 (initial conditions) to the params in one vector
# This is reflected in the model function

function model(du,u,p,t)
    x, y = u
    α, β, δ, γ = p
    du[1] = dx = α*x - β*x*y
    du[2] = dy = -δ*y + γ*x*y
end

function predict_adjoint(u0_params) # Our 1-layer neural network
    prob=ODEProblem(model,u0_params[1:2],(tstart,tend), u0_params[3:end])
    Array(concrete_solve(prob,Tsit5(),u0_params[1:2],u0_params[3:end],saveat=tstart:sampling:tend, abstol=1e-8,reltol=1e-6))
end

function loss_adjoint(u0_params)
    prediction = predict_adjoint(u0_params)
    loss = sum(abs2,prediction - data)
    return loss
end;

function generate_data(u0_params)
    # Generate some data to fit, and add some noise to it
    global data
    data=predict_adjoint(u0_params)
    σN=0.1
    data+=σN*randn(size(data))
    data=abs.(data) #Keep measurements positive
end

function train_model(true_u0_params, init_u0_params)
    # Generate data for each trial. This mutates the global data variable
    generate_data(true_u0_params)
    
    println("The initial loss is $(loss_adjoint(init_u0_params)[1])")
    resinit=DiffEqFlux.sciml_train(loss_adjoint,init_u0_params,ADAM(), maxiters=3000)
    res = DiffEqFlux.sciml_train(loss_adjoint,resinit.minimizer,BFGS(initial_stepnorm = 1e-5))
    println("The learned parameters are $(res.minimizer) with final loss of $(res.minimum)")
    return(res)
end

function train_models_with_params(all_init_u0_params)
    # For Drosophila regulation, we want to learn from two true distributions:
    # Bistable and Mutual inhibition
    # Make the true parameters a parameter to the function as well
    # For now, we will only model 2 protien interaction
    
    # The independent variable is p (I am guessing the saturation ratio of the protien?)
    # From 0.0 to 1.0, 0.0 being no protien and 1.0 being complete saturation
    # For two protiens, we will have p1 and p2, both independent variables
    
    true_params = [1.5,1.0,3.0,1.0]
    
    for params in all_init_u0_params
        # Use the initial condition from the initial_u0_param
        true_u0_params = cat(dims=1, params[1:2], true_params)
        init_u0_params = params
        println("Starting training run with:")
        println("True params $(true_u0_params)")
        println("Initial params $(init_u0_params)")
        model = train_model(true_u0_params, init_u0_params)
        fitPlot(model.minimizer)
        validationPlot(true_u0_params, model.minimizer)
    end
end

# The plot that shows how the generated data compares to the function defined by params
function fitPlot(learned_u0_params)
    tspan=(tstart,tend)
    sol_fit=solve(ODEProblem(model,learned_u0_params[1:2],tspan,learned_u0_params[3:end]), Tsit5())
    tgrid=tstart:sampling:tend
    
    # Plot
    pl=plot(sol_fit, lw=2, legend=false)
    scatter!(pl,tgrid, data[1,:], color=:blue)
    scatter!(pl,tgrid, data[2,:], color=:red)
    xlabel!(pl,"Time")
    ylabel!(pl,"Population")
    title!(pl,"Model Parameter Fits")
    #savefig("Lotka_Volterra_ParamFit.png")
    display(pl)
end

# The plot that shows how the learned parameters compares to data generated from the actual model
function validationPlot(true_u0_params, learned_u0_params)
    tspan=(tstart,tend)
    sol_fit=solve(ODEProblem(model, learned_u0_params[1:2], tspan, learned_u0_params[3:end]), Tsit5())
    sol_actual=solve(ODEProblem(model, true_u0_params[1:2], tspan, true_u0_params[3:end]), Tsit5(), saveat=0.0:0.1:10.0)
    
    # Plot
    pl=scatter(sol_actual, lw=2.0, color=:blue, vars=(0,1))
    scatter!(sol_actual, color=:red, vars=(0,2))
    plot!(pl, sol_fit, lw=2, legend=false, color=:blue, vars=(0,1))
    plot!(pl, sol_fit, lw=2, color=:red, vars=(0,2))
    xlabel!(pl,"Rabbits")
    ylabel!(pl,"Lynx")
    title!(pl,"Validation Plot")
    #savefig("Lotka_Volterra_Validation_Plot.png")
    display(pl)
end

end #module