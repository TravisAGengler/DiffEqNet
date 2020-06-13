module drosophila_regulation

using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots

tstart=0.0
tend=10.0
sampling=0.1
data=0 # Data will be set in generate_data
data_std_dev = 0.02 # Standard deviation on the synthetic data distribution
cur_u0=zeros(0) # This will be populated each training round. This way, we do not "learn" the u0 params

loss_history=zeros(0) # This will be populated each training round and graphed at the end
max_itrs=5000 # The max number of iterations to train for
loss_threshold = 0.001 # If loss doesnt change by this much in loss_threshold_itrs iterations, stop training
loss_threshold_itrs = 100 # The number of iterations to stop at if loss doesnt change by loss_threshold
loss_no_sig_change_itrs = 0 # The number of iterations we have not seen significant change in loss as defined above

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
    if p < 0
        println("pos p: $(p)")
    end
    if k < 0
        println("pos k: $(k)")
    end
    if n < 0
        println("pos n: $(n)")
    end
    p^n/(p^n + k^n)
end

function hill_neg(p, k, n)
    if p < 0
        println("neg p: $(p)")
    end
    if k < 0
        println("neg k: $(k)")
    end
    if n < 0
        println("neg n: $(n)")
    end
    k^n/(p^n + k^n)
end

function gen_reg(p, k, n, φ)
    (hill_pos(p, k, n) * (√(1/2)*cosd(φ*360) + 1/2)) + (hill_neg(p, k, n) * (√(1/2)*sind(φ*360) + 1/2))
end

function model(du,u,p,t)
    # We will be modeling a system of 2 protiens. Systems with more protiens will need to be modeled differently
    # Ignore the diffusion term for now
    # TRICKY: For some reason, p sometimes assume a negative value?
    p1, p2 = abs.(u)
    # Negative values do not make sense in the context of this physical model. Take abs
    α1, α2, β1, β2, k1, k2, n1, n2, φ11, φ12, φ21, φ22 = abs.(p)
    du[1] = α1*gen_reg(p1, k1, n1, φ11)*gen_reg(p2, k2, n2, φ12) - β1*p1 
    du[2] = α2*gen_reg(p1, k1, n1, φ21)*gen_reg(p2, k2, n2, φ22) - β2*p2 
end

function predict_adjoint(params) # Our 1-layer neural network
    prob=ODEProblem(model,cur_u0,(tstart,tend), params)
    Array(concrete_solve(prob,Tsit5(),cur_u0, params,saveat=tstart:sampling:tend, abstol=1e-8,reltol=1e-6))
end

function loss_adjoint(params)
    prediction = predict_adjoint(params)
    loss = sum(abs2, prediction - data)
    return loss
end

function report_params(u0, true_params, params)
    p1, p2 = u0
    α1_t, α2_t, β1_t, β2_t, k1_t, k2_t, n1_t, n2_t, φ11_t, φ12_t, φ21_t, φ22_t = true_params
    α1, α2, β1, β2, k1, k2, n1, n2, φ11, φ12, φ21, φ22 = params
    println("p1: $(p1)")
    println("p2: $(p2)")
    println("α1_true / α1: $(α1_t) / $(α1)")
    println("α2_true / α2: $(α2_t) / $(α2)")
    println("β1_true / β1: $(β1_t) / $(β1)")
    println("β2_true / β2: $(β2_t) / $(β2)")
    println("k1_true / k1: $(k1_t) / $(k1)")
    println("k2_true / k2: $(k2_t) / $(k2)")
    println("n1_true / n1: $(n1_t) / $(n1)")
    println("n2_true / n2: $(n2_t) / $(n2)")
    # These parameters are normalized between 0 and 1    
    println("φ11_true / φ11: $(φ11_t * 360) / $(φ11 * 360)")
    println("φ12_true / φ12: $(φ12_t * 360) / $(φ12 * 360)")
    println("φ21_true / φ21: $(φ21_t * 360) / $(φ21 * 360)")
    println("φ22_true / φ22: $(φ22_t * 360) / $(φ22 * 360)")
end

function generate_data(params)
    # Generate some data to fit, and add some noise to it
    global data
    data=predict_adjoint(params)
    σN=data_std_dev
    data+=σN*randn(size(data))
    data=abs.(data) #Keep measurements positive
end

function generate_init_params(n_params)
    init_params=zeros(n_params)
    for i in eachindex(init_params)
        init_params[i] = 1.0 - rand()
    end
    return init_params
end

function loss_callback(params, loss)
    global loss_no_sig_change_itrs
    
    if size(loss_history)[1] > 0
        loss_delt = abs.(last(loss_history) - loss)
    else
        loss_delt = 0
    end
    
    append!(loss_history, loss)
    
    if loss_delt > loss_threshold
        loss_no_sig_change_itrs = 0
    else
        loss_no_sig_change_itrs += 1
        if loss_no_sig_change_itrs >= loss_threshold_itrs
            return true
        end
    end
    return false
end

function train_model(true_params, init_params)
    # Reset our loss history
    global loss_history
    loss_history = zeros(0)
    
    # Train the model
    res=DiffEqFlux.sciml_train(loss_adjoint, init_params, ADAM(), maxiters=max_itrs, cb=loss_callback)
    return(res)
end

function train_model_with_params(all_u0, true_params)
    global cur_u0
    global loss_no_sig_change_itrs
    
    init_params = generate_init_params(12)
    true_and_learned_params =  Any[]
    for u0 in all_u0
        cur_u0 = u0
        loss_no_sig_change_itrs = 0
        # Generate synthetic data for each trial run
        generate_data(true_params)
        
        # Report initial conditions and parameters
        println("Initial loss: $(loss_adjoint(init_params)[1])")
        
        # Train model
        model = train_model(true_params, init_params)
        # The model can learn negative parameters, but our calculations account for that.
        # Use the abs of the params here for our reporting
        learned_params = abs.(model.minimizer)
        
        # Report learned parameters
        println("Finished training after $(size(loss_history)[1]) iterations")
        println("Learned params")
        report_params(cur_u0, true_params, learned_params)
        println("Learned loss: $(loss_adjoint(learned_params)[1])")
        
        # Make plots
        dataPlot(learned_params)
        validationPlot(true_params, learned_params)
        lossPlot(loss_history)
        push!(true_and_learned_params, [true_params, learned_params])
    end
    phasePlot(all_u0, true_and_learned_params)
end

function train_bistable_model_with_params(all_u0)
    println("Learning bistable parameters")
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
    train_model_with_params(all_u0, true_params_bistable)
end


function train_mutual_inhib_model_with_params(all_u0)   
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
    train_model_with_params(all_u0, true_params_mutual)
end

# The plot that shows how the generated data compares to the function defined by params
# The solid line is the actual model
# The scatter plot is the generated data
function dataPlot(learned_params)
    tspan=(tstart,tend)
    sol_learned=solve(ODEProblem(model,cur_u0,tspan,learned_params), Tsit5())
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
function validationPlot(true_params, learned_params)
    tspan=(tstart,tend)
    sol_learned=solve(ODEProblem(model, cur_u0, tspan, learned_params), Tsit5())
    sol_actual=solve(ODEProblem(model, cur_u0, tspan, true_params), Tsit5())
    
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
    last_x = size(loss_history)[1]-1
    x = 0:last_x
    pl = plot(x, loss_history, lw=2, legend=false, color=:red)
    xlabel!(pl,"Iteration")
    ylabel!(pl,"Training Loss")
    title!(pl,"Loss Plot")
    display(pl)
end

# The plot that shows the phase dynamics between the two variables p1 and p2 for multiple initial conditions
function phasePlot(all_u0, true_and_learned_params)
    colors=[:blue, :red, :green, :purple, :black, :cyan, :orange, :gray]
    tspan=(tstart,tend)
    pl = nothing
    global cur_u0
    for i in eachindex(true_and_learned_params)
        cur_u0 = all_u0[i]
        cl = colors[i]
        tr = true_and_learned_params[i][1]
        ln = true_and_learned_params[i][2]
        sol_actual=solve(ODEProblem(model, cur_u0, tspan, tr), Tsit5(), saveat=0.0:0.01:10.0)
        sol_learned=solve(ODEProblem(model, cur_u0, tspan, ln), Tsit5(), saveat=0.0:0.01:10.0)
        
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
