# make sure local environment is activated
using Pkg
Pkg.activate(".")

using CSV
using Plots
using MLDataUtils

using Distributed
using DataFrames

# Add workers
# nprocs() == 1 && addprocs(10, exeflags=["--project=$(Base.active_project())"])
# workers()

using AutoMLPipeline
using Random

using Hyperopt
using Plots
using StatsPlots


# Load the data
df_red = CSV.read("winequality-red.csv", DataFrame)
# names = CSV.read("winequality.names", DataFrame)

# split the data into training and testing
# Random.seed!(42)
rng = MersenneTwister(1234)
train, test = splitobs(shuffleobs(df_red, rng=rng), at=0.7)
X_train = train[:, 1:end-1]
Y_train = train[:, end] |> Vector
X_test = test[:, 1:end-1]
Y_test = test[:, end] |> Vector
# head(x)=first(x,5)
# head(df_red)

# Define the model
#### Learners
rf = hp -> SKLearner("RandomForestClassifier", n_estimators=hp[1], max_depth=hp[2], max_features=hp[3], random_state=0)
gb = hp -> SKLearner("GradientBoostingClassifier", n_estimators=Int(hp[1]), learning_rate=hp[2], max_depth=Int(hp[3]), random_state=0)
svc = hp -> SKLearner("SVC", C=hp[1], kernel=hp[2], degree=hp[3], random_state=0)

#### Decomposition
pca = SKPreprocessor("PCA", Dict(:n_components => 5, :random_state => 0))
ica = SKPreprocessor("FastICA", Dict(:n_components => 5, :whiten => true))

#### Scaler 
rb = SKPreprocessor("RobustScaler")
pt = SKPreprocessor("PowerTransformer")
norm = SKPreprocessor("Normalizer")
mx = SKPreprocessor("MinMaxScaler")
std = SKPreprocessor("StandardScaler")

#### categorical preprocessing
ohe = OneHotEncoder()

#### Column selector
catf = CatFeatureSelector()
numf = NumFeatureSelector()
disc = CatNumDiscriminator()


Random.seed!(42)
# first transform the data
# use OneHotEncoder for categorical data and RobustScaler for numerical data
# but technically, there should be no categorical data in this dataset
pl = @pipeline disc |> (catf |> ohe) + (numf |> rb |> pca)
X_train_trans = AutoMLPipeline.fit_transform!(pl, X_train, Y_train)
X_test_trans = AutoMLPipeline.transform(pl, X_test)

# Define the pipeline
function HPOLearner(learner, X, Y)
    # we will use accuracy as the metric
    # we will use 5 fold cross validation
    Random.seed!(42)
    mean, sd, _ = crossvalidate(learner, X, Y, "accuracy_score", nfolds=5, verbose=false)
    # weight the accuracy by inverse variance
    # negative for minimization
    # return -mean / sd^2
    return -mean
end

function plotHyperopt(ho, figname)
    params = hcat(ho.history...)'
    vals = [t[1] for t in ho.results]

    p1 = scatter(params[:, 1], params[:, 2], zcolor=-vals, xlabel="n_estimators", ylabel="max_depth")
    plot!(p1, colorbar_title="Accuracy", colorbar=true, legend=false)
    p2 = scatter(params[:, 2], params[:, 3], zcolor=-vals, xlabel="max_depth", ylabel="max_feature")
    plot!(p2, colorbar_title="Accuracy", colorbar=true, legend=false)
    p3 = scatter(params[:, 1], params[:, 3], zcolor=-vals, xlabel="n_estimators", ylabel="max_feature")
    plot!(p3, colorbar_title="Accuracy", colorbar=true, legend=false)
    p_empty = plot(legend=false, ticks=nothing, border=:none)
    p = plot(p1, p_empty, p3, p2, size=(800, 600), layout=(2, 2))
    savefig(p, figname)
end

function writeToCSV(filename, ho, colnames)
    params = permutedims(hcat(ho.history...))
    vals = [t[1] for t in ho.results]
    CSV.write(filename, DataFrame(hcat(-vals, params), colnames))
end

function writeTestResult(dir, ho_params_name, sampler_name, all_ho, all_cv)
    df_cv = DataFrame(all_cv)
    df_ho = DataFrame(permutedims(hcat([ho.minimizer for ho in all_ho]...)), ho_params_name[2:end])
    CSV.write(dir * "all_cv.csv", df_cv)
    CSV.write(dir * "all_ho_params.csv", df_ho)
    for i in 1:length(all_ho)
        writeToCSV(dir * sampler_name[i] * ".csv", all_ho[i], ho_params_name)
    end
end

function analyzePerf(model, dir, ho_params_name, sampler_name, all_ho, X_test, Y_test)
    Random.seed!(42)
    cv_res = []
    # total_mean = []
    p = boxplot()
    perfplot = plot()
    for (i, ho) in enumerate(all_ho)
        # plotHyperopt(ho, dir * sampler_name[i])
        push!(cv_res, crossvalidate(model(ho.minimizer), X_test, Y_test, "accuracy_score", nfolds=5, verbose=true))
        scores = [t[1] for t in ho.results]
        scores = scores[.!isnan.(scores)]
        # push!(total_mean, scores[.!isnan(scores)])
        boxplot!(p, [sampler_name[i]], -scores, ylabel="Accuracy", legend=false)
        plot!(perfplot, -scores, label=sampler_name[i], ylabel="Accuracy", legend=true)
    end
    # println(cv_res)
    plot!(p, legend=false, ylabel="Accuracy score")
    savefig(p, dir * "boxplot.png")
    savefig(perfplot, dir * "perfplot.png")
    writeTestResult(dir, ho_params_name, sampler_name, all_ho, cv_res)
end

# For Random Forest
println("Random Forest")
# Define hypterparameter function
HPO_rf = hp -> HPOLearner(rf(hp), X_train_trans, Y_train)
x0 = [100, 10, 10]
# Hyperparameter bounds for n_estimators, max_depth, max_feature
lower = [10, 1, 1]
upper = [300, 30, 30]
# GA
# using Evolutionary
# res = Evolutionary.optimize(HPO_rf, BoxConstraints(lower, upper), x0,
#                         GA(populationSize=100, crossoverRate=0.8, mutationRate=0.4),
#                         Evolutionary.Options(reltol=1e-4, iterations=10, show_trace=true, parallelization=:thread))
# Cannot use EA here, because inputs are all integer. Use random search instead

# random search
println("Random Search")
hors = @time @hyperopt for i = 50,
    sampler = RandomSampler(rng), # This is default if none provided
    n_est = lower[1]:upper[1],
    max_depth = lower[2]:upper[2],
    max_feature = lower[3]:upper[3]
    # print(i, "\t", n_est, "\t", max_depth, "\t", max_feature, "   \t")
    HPO_rf([n_est, max_depth, max_feature]), [n_est, max_depth, max_feature]
end

# use Hyperband for optimization
println("Hyperband")
hohb = @time @hyperopt for i = 50,
    sampler = Hyperband(R=50, η=3, inner=RandomSampler(rng)),
    n_est = lower[1]:upper[1],
    max_depth = lower[2]:upper[2],
    max_feature = lower[3]:upper[3]

    if state !== nothing
        n_est, max_depth, max_feature = state
    end
    # println(i, "\t", n_est, "\t", max_depth, "\t", max_feature, "   \n")
    # res = Optim.optimize(x -> HPO_rf(round.(Int, x)), [n_est, max_depth, max_feature],
    #       float(lower), float(upper), NelderMead(), Optim.Options(f_calls_limit=round(Int, i)+1))
    # @show Optim.minimum(res), round.(Int, Optim.minimizer(res))
    HPO_rf([n_est, max_depth, max_feature]), [n_est, max_depth, max_feature]
end

# Hyperband with Bayesian optimization
println("Hyperband with Bayesian optimization")
hohbbo = @time @hyperopt for i = 50,
    sampler = Hyperband(R=50, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()], 
                        random_sampler=RandomSampler(rng))),
    n_est = lower[1]:upper[1],
    max_depth = lower[2]:upper[2],
    max_feature = lower[3]:upper[3]

    if state !== nothing
        n_est, max_depth, max_feature = round.(Int, abs.(state))
    end
    # print(i, "\n")
    if !all(lower .<= [n_est, max_depth, max_feature] .<= upper)
        # to prevent type error and circumvent bug in their code
        # the bug will step out of the search space
        NaN, [n_est, max_depth, max_feature]
    else
        HPO_rf([n_est, max_depth, max_feature]), [n_est, max_depth, max_feature]
    end
end

analyzePerf(rf, "./rf/", ["accuracy", "n_estimators", "max_depth", "max_feature"],
    ["random_search", "hyperband_rs", "hyperband_bo"],
    [hors, hohb, hohbbo], X_test_trans, Y_test)

# For Gradient Boosting
println("Gradient Boosting")
# Define hypterparameter function
HPO_gb = hp -> HPOLearner(gb(hp), X_train_trans, Y_train)
n_est_range = 10:500
lr_range = LinRange(0.01, 0.5, 100)
max_depth_range = 1:30

# random search
println("Random Search")
hors = @time @hyperopt for i = 50,
    sampler = RandomSampler(rng), # This is default if none provided
    n_est = n_est_range,
    lr = lr_range,
    max_depth = max_depth_range
    # print(i, "\t", n_est, "\t", max_depth, "\t", max_feature, "   \t")
    params = [n_est, lr, max_depth]
    HPO_gb(params), params
end

# use Hyperband for optimization
println("Hyperband")
hohb = @time @hyperopt for i = 50,
    sampler = Hyperband(R=50, η=3, inner=RandomSampler(rng)),
    n_est = n_est_range,
    lr = lr_range,
    max_depth = max_depth_range

    if state !== nothing
        n_est, lr, max_depth = state
    end
    @show HPO_gb([n_est, lr, max_depth]), [n_est, lr, max_depth]
end

# Hyperband with Bayesian optimization
println("Hyperband with Bayesian optimization")
hohbbo = @time @hyperopt for i = 50,
    sampler = Hyperband(R=50, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()],
                        random_sampler=RandomSampler(rng))),
    n_est = n_est_range,
    lr = lr_range,
    max_depth = max_depth_range

    if state !== nothing
        n_est, lr, max_depth = state
    end
    if !(n_est in n_est_range) || !(lr in lr_range) || !(max_depth in max_depth_range)
        # to prevent type error and circumvent bug in their code
        # the bug will step out of the search space
        @show NaN, [n_est, lr, max_depth]
    else
        @show HPO_gb([n_est, lr, max_depth]), [n_est, lr, max_depth]
    end
end

analyzePerf(gb, "./gb/", ["accuracy", "n_estimators", "learning_rate", "max_depth"],
    ["random_search", "hyperband_rs", "hyperband_bo"],
    [hors, hohb, hohbbo], X_test_trans, Y_test)


# For SVC
println("SVM")
# Define hypterparameter function
HPO_svc = hp -> HPOLearner(svc(hp), X_train_trans, Y_train)
C_range = LinRange(0.1, 10, 100)
kernel_range = ["linear", "poly", "rbf", "sigmoid"]
degree_range = 1:10
# random search
println("Random Search")
hors = @time @hyperopt for i = 50,
    sampler = RandomSampler(rng),
    C = C_range,
    kernel = kernel_range,
    degree = degree_range
    params = [C, kernel, degree]
    HPO_svc(params), params
end

# use Hyperband for optimization
println("Hyperband")
hohb = @time @hyperopt for i = 50,
    sampler = Hyperband(R=50, η=3, inner=RandomSampler(rng)),
    C = C_range,
    kernel = kernel_range,
    degree = degree_range

    if state !== nothing
        C, kernel, degree = state
    end
    @show HPO_svc([C, kernel, degree]), [C, kernel, degree]
end

# Cannot use hyperband with BO with categorical variables given
# the limit of the package

analyzePerf(svc, "./svm/", ["accuracy", "C", "kernel", "degree"],
    ["random_search", "hyperband_rs"],
    [hors, hohb], X_test_trans, Y_test)


# Compare performance for all models
df_rf = CSV.read("./rf/hyperband_rs.csv", DataFrame)
df_gb = CSV.read("./gb/hyperband_rs.csv", DataFrame)
df_svm = CSV.read("./svm/hyperband_rs.csv", DataFrame)
p = boxplot(["Random Forest"], df_rf.accuracy, ylabel="Accuracy", legend=false)
boxplot!(p, ["Gradient Boosting"], df_gb.accuracy, ylabel="Accuracy", legend=false)
boxplot!(p, ["SVM"], df_svm.accuracy, ylabel="Accuracy", legend=false)
savefig(p, "all_hbrs_perf_boxplot.png")