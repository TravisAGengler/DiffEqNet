module drosophila_regulation

using DifferentialEquations, Flux, Optim, DiffEqFlux, Plots

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

# These parameters dictate how many points are generated in synthetic data
# and how many points are solved in the ODE_Problem
# n points = tend-tstart / sampling
tstart=0.0
tend=10.0
sampling=0.01

all_u0=[] # All initial starting parameters
u0=[] # This will be populated each training round. This way, we do not "learn" the u0 params
all_data=[] # Data will be set in generate_data
data_std_dev = 0.02 # Standard deviation on the synthetic data distribution

loss_history=[] # This will be populated each training round and graphed at the end
itrs=0 # The number of iterations we have been training for
max_itrs=500 # The max number of iterations to train for
loss_no_decrease_max = 100 # The number of iterations to stop at if loss doesnt decrease
loss_no_decrease = 0 # The number of iterations we have not had loss decrease
loss_delt_threshold_percent=0.10 # The minimum percent change required to count as "changed" in loss

# These will be populated each training round and graphed at the end
φ11_history=[]
φ12_history=[]
φ21_history=[]
φ22_history=[]

# The true parameters for a system of two proteins with a bistable relationship
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

# The true parameters for a system of two proteins with a relationship of mutual inhibition
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
    # Negative values do not make sense in the context of this physical model. Take abs
    # TRICKY: For some reason, p sometimes assumes a negative value?
    p1, p2 = abs.(u)
    α1, α2, β1, β2, k1, k2, n1, n2, φ11, φ12, φ21, φ22 = abs.(p)

     # Ignore the diffusion term for now
    du[1] = α1*gen_reg(p1, k1, n1, φ11)*gen_reg(p2, k2, n2, φ12) - β1*p1 
    du[2] = α2*gen_reg(p1, k1, n1, φ21)*gen_reg(p2, k2, n2, φ22) - β2*p2 
end

function predict(params)
    prob = ODEProblem(model, u0, (tstart,tend), params)
    Array(concrete_solve(prob, Tsit5(), u0, params, saveat=tstart:sampling:tend, abstol=1e-8, reltol=1e-6, alg_hints=[:stiff], force_dtmin=true))
end

function loss(params)
    global u0
    total_loss = 0
    for i in eachindex(all_u0)
        u0 = all_u0[i]
        data = all_data[i]
        prediction = predict(params)
        total_loss += sum(abs2, prediction - data)
    end
    return total_loss
end

function generate_data(n, params)
    global all_data
    global u0
    for i in 1:n
        u0 = all_u0[i]
        # Generate some data to fit, and add some noise to it
        data=predict(params)
        σN=data_std_dev
        data+=σN*randn(size(data))
        data=abs.(data) #Keep measurements positive
        push!(all_data, data)
    end
end

function generate_init_conditions(n)
    global all_u0
    for _ in 1:n
        push!(all_u0, [1.0 - rand(), 1.0 - rand()])
    end
end

function generate_init_params(n)
    init_params = zeros(0)
    for _ in 1:n
        push!(init_params, 1.0 - rand())
    end
    return init_params
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

# This callback has two purposes. Early stopping and recording history of parameters
function loss_callback(params, loss)
    global loss_no_decrease
    global itrs
    
    itrs+=1
    
    if size(loss_history)[1] > 0
        last_loss = last(loss_history)
    else
        last_loss = 0
    end
    
    loss_delt = loss - last_loss
    
    append!(loss_history, loss)
    append!(φ11_history, params[9] * 360)
    append!(φ12_history, params[10] * 360)
    append!(φ21_history, params[11] * 360)
    append!(φ22_history, params[12] * 360)
    
    if loss_delt < 0 && abs.(loss_delt) >= last_loss*loss_delt_threshold_percent
        loss_no_decrease = 0
    else
        loss_no_decrease += 1
        if loss_no_decrease >= loss_no_decrease_max
            println("Training halted after no significant loss change in $(loss_no_decrease_max) iterations")
            return true
        end
    end
    
    if itrs >= max_itrs
        println("Training halted after exceeding $(max_itrs) maximum iterations")
        return true
    end
    return false
end

function train_model(init_params)
    # Train the model
    # TRICKY: It looks like the maxitrs parameter is overridden by the loss_callback
    # It looks like when mixing the stopping methods, it will RESTART training when reaching maxiters!
    # The argument is required, so lets just set it to a value it should never reach
    opt=AdaMax(0.005, (0.9, 0.8))
    res=DiffEqFlux.sciml_train(loss, init_params, opt, maxiters= max_itrs+1, cb=loss_callback)
    return(res)
end

function reset_globals()
    global all_u0
    global all_data
    global loss_no_decrease
    global itrs
    global loss_history
    global φ11_history
    global φ12_history
    global φ21_history
    global φ22_history
    
    all_u0=[] 
    all_data=[] 
    loss_no_decrease = 0
    itrs = 0
    loss_history=[] 
    φ11_history=[]
    φ12_history=[]
    φ21_history=[]
    φ22_history=[]
end

function train_model_with_params(n, true_params)   
    # Reset globals before each run
    reset_globals()
    
    # Generate starting conditions, random initial parameters and true data
    init_params = generate_init_params(12)
    generate_init_conditions(n)
    generate_data(n, true_params)
        
    # Report initial conditions and parameters
    println("Initial loss: $(loss(init_params))")

    # Train model
    model = train_model(init_params)
    # The model can learn negative parameters, but our calculations account for that.
    # Use the abs of the params here for our reporting
    learned_params = abs.(model.minimizer)

    # Report learned parameters
    println("Finished training after $(itrs) iterations")
    #println("Learned params")
    #report_params(u0, true_params, learned_params)
    println("Learned loss: $(loss(learned_params)[1])")

    # Make plots
    dataPlot(learned_params)
    validationPlot(true_params, learned_params)
    lossPlot(loss_history)
    phi_plot(φ11_history, "11", true_params[9] * 360)
    phi_plot(φ12_history, "12", true_params[10] * 360)
    phi_plot(φ21_history, "21", true_params[11] * 360)
    phi_plot(φ22_history, "22", true_params[12] * 360)
    phasePlot(true_params, learned_params)
end

function train_mutual_bistable(n)
    train_model_with_params(n, true_params_bistable)
end

function train_mutual_inhib(n)
    train_model_with_params(n, true_params_bistable)
end

# The plot that shows how the generated data compares to the function defined by params
# The solid line is the actual model
# The scatter plot is the generated data
function dataPlot(learned_params)
    for i in eachindex(all_u0)
        u0 = all_u0[i]
        data = all_data[i]
        tspan=(tstart,tend)
        sol_learned=solve(ODEProblem(model,u0,tspan,learned_params), Tsit5())
        tgrid=tstart:sampling:tend

        # Plot
        pl=plot(sol_learned, lw=2, color=:blue, vars=(0,1), label="Learned model p1")
        plot!(pl, sol_learned, lw=2, color=:red, vars=(0,2), label="Learned model p2")
        scatter!(pl,tgrid, data[1,:], color=:blue, label="Generated Data p1")
        scatter!(pl,tgrid, data[2,:], color=:red,  label="Generated Data p2")
        xlabel!(pl,"Time")
        ylabel!(pl,"Protien Expression")
        p1_round = round(u0[1], digits=2)
        p2_round = round(u0[2], digits=2)
        title!(pl,"Data Fit with Learned Model. Init p1:$(p1_round) p2:$(p2_round)")
        display(pl)
    end
end

# The plot that shows how the learned parameters compares to data generated from the actual model
# The solid line is the actual model
# The scatter line is the learned model
function validationPlot(true_params, learned_params)
    for i in eachindex(all_u0)
        u0 = all_u0[i]
        tspan=(tstart,tend)
        sol_learned=solve(ODEProblem(model, u0, tspan, learned_params), Tsit5())
        sol_actual=solve(ODEProblem(model, u0, tspan, true_params), Tsit5())

        # Plot
        pl = plot(sol_actual, lw=2, color=:blue, linestyle = :dot, vars=(0,1), label="True model p1")
        plot!(pl, sol_actual, lw=2, color=:red, linestyle = :dot, vars=(0,2), label="True model p2")
        plot!(pl, sol_learned, lw=2, color=:blue, vars=(0,1), label="Learned model p1")
        plot!(pl, sol_learned, lw=2, color=:red, vars=(0,2), label="Learned model p2")
        xlabel!(pl,"Time")
        ylabel!(pl,"Protien Expression")
        p1_round = round(u0[1], digits=2)
        p2_round = round(u0[2], digits=2)
        title!(pl,"Validation Plot Init p1:$(p1_round) p2:$(p2_round)")
        display(pl)
    end
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
function phasePlot(true_params, learned_params)
    colors=[:blue, :red, :green, :purple, :black, :cyan, :orange, :gray]
    tspan=(tstart,tend)
    pl = nothing
    global u0
    for i in eachindex(all_u0)
        u0 = all_u0[i]
        cl = colors[mod(i, size(colors)[1])+1]
        
        sol_actual=solve(ODEProblem(model, u0, tspan, true_params), Tsit5(), saveat=tstart:sampling:tend)
        sol_learned=solve(ODEProblem(model, u0, tspan, learned_params), Tsit5(), saveat=tstart:sampling:tend)
        
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
        
        p1_round_a = round(p1_actual[1], digits=2)
        p2_round_a = round(p2_actual[1], digits=2)
        if(isnothing(pl))
            pl = plot(p1_actual, p2_actual, lw=2, color=cl, linestyle = :dot, label="True Init p1:$(p1_round_a[1]) p2:$(p2_round_a[1])")
        else
            plot!(pl, p1_actual, p2_actual, lw=2, color=cl, linestyle = :dot, label="True Init p1:$(p1_round_a[1]) p2:$(p2_round_a[1])")
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

function phi_plot(φ_history, subscript, expected)
    pl = histogram(φ_history,bins=0:5:360, label="", normalize=:probability)
    scatter!(pl, [expected], [0], label="Expected Phi")
    learned = mod(last(φ_history), 360)
    scatter!(pl, [learned], [0], label="Learned Phi")
    plot!(pl, [cosd], 0:5:360, label="cos(Phi)") 
    xlabel!(pl,"Phi")
    ylabel!(pl,"Frequency")
    diff = round(abs(cosd(expected)-cosd(learned)), digits=2)
    title!(pl,"Phi $(subscript) history. Diff: $(diff)")
    display(pl)
end

end #module
