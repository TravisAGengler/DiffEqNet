module drosophila_regulation

using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots

tstart=0.0
tend=10.0
sampling=0.1
data=0 # Data will be set in generate_data
loss_history=zeros(0) # This will be populated each training round and graphed at the end
max_itrs=500 # The number of iterations to train for

# u0_params in this file refers to the concatination of the u0 (initial conditions) to the params in one vector
# This is reflected in the model function

# For Drosophila protien regulation, we want to learn from two true distributions:
# Bistable and Mutual inhibition
# Make the true parameters a parameter to the function as well
# For now, we will only model 2 protien interaction

# The independent variable is p (Expression level of protein)
# From 0.0 to 1.0, 0.0 being no protien and 1.0 being complete expression
# For two protiens, we will have p1 and p2, both independent variables

# For each protien, there are α (synthesis) and β (decay) parameters 
# The general form of the differential equation includes n φ parameters for each of the n protiens
# k and n parameters control the hill function

function hill_pos(p, k, n)
    p^n/(p^n + k^n)
end

function hill_neg(p, k, n)
    k^n/(p^n + k^n)
end

function gen_reg(p, k, n, φ)
    (hill_pos(p, k, n) * (√(1/2)*cosd(φ*360) + 1/2)) + (hill_neg(p, k, n) * (√(1/2)*sind(φ*360) + 1/2))
end

function model(du,u,p,t)
    # We will be modeling a system of 2 protiens. Systems with more protiens will need to be modeled differently
    # Ignore the diffusion term for now
    p1, p2 = u
    α1, α2, β1, β2, k1, k2, n1, n2, φ11, φ12, φ21, φ22 = p
    du[1] = α1*gen_reg(p1, k1, n1, φ11)*gen_reg(p2, k2, n2, φ12) - β1*p1 
    du[2] = α2*gen_reg(p1, k1, n1, φ21)*gen_reg(p2, k2, n2, φ22) - β2*p2 
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
    σN=0.05
    data+=σN*randn(size(data))
    data=abs.(data) #Keep measurements positive
end

function loss_callback(u0_params, loss)
    append!(loss_history, loss)
    return false
end

function train_model(true_u0_params, init_u0_params)
    # Generate data for each trial. This mutates the global data variable
    generate_data(true_u0_params)
    
    global loss_history
    loss_history = zeros(0)
    
    println("The initial loss is $(loss_adjoint(init_u0_params)[1])")
    global max_itrs
    res=DiffEqFlux.sciml_train(loss_adjoint,init_u0_params,ADAM(), maxiters=max_itrs, cb=loss_callback)
    println("The learned parameters are $(res.minimizer) with final loss of $(res.minimum)")
    return(res)
end

function train_bistable_model_with_params(all_init_u0_params)    
    true_params_bistable = 
       [0.29,           # α1
        0.19,           # α2
        0.29,           # β1
        0.19,           # β2
        0.11,           # k1
        0.08,           # k2
        2.0,            # n1
        2.0,            # n2
        315.0 / 360.0,  # φ11
        135.0 / 360.0,  # φ12
        135.0 / 360.0,  # φ21
        315.0 / 360.0]  # φ22
    
    true_and_learned_u0_params =  Any[]
    for params in all_init_u0_params
        println("Learning bistable parameters")
        # Use the initial condition from the initial_u0_param
        true_u0_params = cat(dims=1, params[1:2], true_params_bistable)
        init_u0_params = params
        println("Starting training run with:")
        println("True params $(true_u0_params)")
        println("Initial params $(init_u0_params)")
        model = train_model(true_u0_params, init_u0_params)
        learned_u0_params = model.minimizer
        dataPlot(learned_u0_params)
        validationPlot(true_u0_params, learned_u0_params)
        lossPlot(loss_history)
        push!(true_and_learned_u0_params, [true_u0_params, learned_u0_params])
    end
    phasePlot(true_and_learned_u0_params)
end


function train_mutual_inhib_model_with_params(all_init_u0_params)   
    true_params_mutual = 
       [0.29,           # α1
        0.19,           # α2
        0.26,           # β1
        0.19,           # β2
        0.11,           # k1
        0.12,           # k2
        2.0,            # n1
        2.0,            # n2
        45.0  / 360.0,  # φ11
        135.0 / 360.0,  # φ12
        135.0 / 360.0,  # φ21
        45.0  / 360.0]  # φ22

    true_and_learned_u0_params = Any[]
    for params in all_init_u0_params
        println("Learning mutual inhibition parameters")
        # Use the initial condition from the initial_u0_param
        true_u0_params = cat(dims=1, params[1:2], true_params_mutual)
        init_u0_params = params
        println("Starting training run with:")
        println("True params $(true_u0_params)")
        println("Initial params $(init_u0_params)")
        model = train_model(true_u0_params, init_u0_params)
        learned_u0_params = model.minimizer
        dataPlot(learned_u0_params)
        validationPlot(true_u0_params, learned_u0_params)
        lossPlot(loss_history)
        push!(true_and_learned_u0_params, [true_u0_params, learned_u0_params])
    end
    phasePlot(true_and_learned_u0_params)
end


# The plot that shows how the generated data compares to the function defined by params
# The solid line is the actual model
# The scatter plot is the generated data
function dataPlot(learned_u0_params)
    tspan=(tstart,tend)
    sol_learned=solve(ODEProblem(model,learned_u0_params[1:2],tspan,learned_u0_params[3:end]), Tsit5())
    tgrid=tstart:sampling:tend
    
    # Plot
    pl=plot(sol_learned, lw=2, color=:blue, vars=(0,1), label="Learned model p1")
    plot!(pl, sol_learned, lw=2, color=:red, vars=(0,2), label="Learned model p2")
    scatter!(pl,tgrid, data[1,:], color=:blue, label="Generated Data p1")
    scatter!(pl,tgrid, data[2,:], color=:red,  label="Generated Data p2")
    xlabel!(pl,"Time")
    ylabel!(pl,"Protien Expression")
    title!(pl,"Data Fit with Learned Model")
    display(pl)
end

# The plot that shows how the learned parameters compares to data generated from the actual model
# The solid line is the actual model
# The scatter line is the learned model
function validationPlot(true_u0_params, learned_u0_params)
    tspan=(tstart,tend)
    sol_learned=solve(ODEProblem(model, learned_u0_params[1:2], tspan, learned_u0_params[3:end]), Tsit5())
    sol_actual=solve(ODEProblem(model, true_u0_params[1:2], tspan, true_u0_params[3:end]), Tsit5())
    
    # Plot
    pl = plot(sol_actual, lw=2, color=:blue, linestyle = :dot, vars=(0,1), label="True model p1")
    plot!(pl, sol_actual, lw=2, color=:red, linestyle = :dot, vars=(0,2), label="True model p2")
    plot!(pl, sol_learned, lw=2, color=:blue, vars=(0,1), label="Learned model p1")
    plot!(pl, sol_learned, lw=2, color=:red, vars=(0,2), label="Learned model p2")
    xlabel!(pl,"Time")
    ylabel!(pl,"Protien Expression")
    title!(pl,"Validation Plot")
    display(pl)
end

# The plot that shows how the loss changed over the learning iterations
function lossPlot(loss_history)    
    # Plot
    x = 0:max_itrs
    pl = plot(x, loss_history, lw=2, legend=false, color=:red)
    xlabel!(pl,"Iteration")
    ylabel!(pl,"Training Loss")
    title!(pl,"Loss Plot")
    display(pl)
end

# The plot that shows the phase dynamics between the two variables p1 and p2 for multiple initial conditions
function phasePlot(true_and_learned_u0_params)
    colors=[:blue, :red, :green, :purple, :black, :cyan, :orange, :gray]
    tspan=(tstart,tend)
    pl = nothing
    for i in eachindex(true_and_learned_u0_params)
        cl = colors[i]
        tr = true_and_learned_u0_params[i][1]
        ln = true_and_learned_u0_params[i][2]
        sol_actual=solve(ODEProblem(model, tr[1:2], tspan, tr[3:end]), Tsit5(), saveat=0.0:0.01:10.0)
        sol_learned=solve(ODEProblem(model, ln[1:2], tspan, ln[3:end]), Tsit5(), saveat=0.0:0.01:10.0)
        
        p1_actual = zeros(0)
        p2_actual = zeros(0)
        for j in eachindex(sol_actual.u)
            append!(p1_actual, sol_actual[j][1])
            append!(p2_actual, sol_actual[j][2])
        end

        p1_learned = zeros(0)
        p2_learned = zeros(0)
        for j in eachindex(sol_learned.u)
            append!(p1_learned, sol_learned[j][1])
            append!(p2_learned, sol_learned[j][2])
        end
        
        if(isnothing(pl))
            pl = plot(p1_actual, p2_actual, lw=2, color=cl, linestyle = :dot, label="True Init p1:$(p1_actual[1]) p2:$(p2_actual[1])")
        else
            plot!(pl, p1_actual, p2_actual, lw=2, color=cl, linestyle = :dot, label="True Init p1:$(p1_actual[1]) p2:$(p2_actual[1])")
        end
        p1_round = round(p1_learned[1], digits=2)
        p2_round = round(p2_learned[1], digits=2)
        plot!(pl, p1_learned, p2_learned, lw=2, color=cl, label="Predicted Init p1:$(p1_round) p2:$(p2_round)") 
    end
    # Plot
    # p1 is x axis 
    # p2 is y axis 
    xlabel!(pl,"P1(t)")
    ylabel!(pl,"P2(t)")
    title!(pl,"Phase Dynamics P1(t)P2(t)")
    display(pl)
end

end #module
