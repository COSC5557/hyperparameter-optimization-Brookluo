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
train, test = splitobs(shuffleobs(df_red, rng=rng), at = 0.7)
X_train = train[:, 1:end-1] 
Y_train = train[:,end] |> Vector
X_test = test[:, 1:end-1]
Y_test = test[:, end] |> Vector
# head(x)=first(x,5)
# head(df_red)

# Define the model
#### Learners
rf       = hp -> SKLearner("RandomForestClassifier", n_estimators=hp[1], max_depth=hp[2], max_features=hp[3], random_state=0)
gb       = hp -> SKLearner("GradientBoostingClassifier", n_estimators=Int(hp[1]), learning_rate=hp[2], max_depth=Int(hp[3]), random_state=0)
svc      = hp -> SKLearner("SVC", C=hp[1], kernel=hp[2], degree=hp[3], random_state=0)
logistic = hp -> SKLearner("LogisticRegression", C=hp[1], random_state=0)

#### Decomposition
pca = SKPreprocessor("PCA", Dict(:n_components=>5, :random_state=>0))
ica = SKPreprocessor("FastICA", Dict(:n_components=>5, :whiten=>true))

#### Scaler 
rb   = SKPreprocessor("RobustScaler")
pt   = SKPreprocessor("PowerTransformer")
norm = SKPreprocessor("Normalizer")
mx   = SKPreprocessor("MinMaxScaler")
std  = SKPreprocessor("StandardScaler")

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


# For Random Forest
println("Random Forest")
# Define hypterparameter function
HPO_rf = hp -> HPOLearner(rf(hp), X_train_trans, Y_train)
# Cannot use EA here, because inputs are all integer. Use random search instead
# Random.seed!(42)
x0 = [100, 10, 10]
lower = [10, 1, 1]
upper = [300, 30, 30]
# GA
# using Evolutionary
# res = Evolutionary.optimize(HPO_rf, BoxConstraints(lower, upper), x0,
#                         GA(populationSize=100, crossoverRate=0.8, mutationRate=0.4),
#                         Evolutionary.Options(reltol=1e-4, iterations=10, show_trace=true, parallelization=:thread))

# random search
println("Random Search")
hors = @time @hyperopt for i=50,
        sampler = RandomSampler(rng), # This is default if none provided
        n_est = lower[1]:upper[1],
        max_depth = lower[2]:upper[2],
        max_feature = lower[3]:upper[3]
    # print(i, "\t", n_est, "\t", max_depth, "\t", max_feature, "   \t")
    HPO_rf([n_est, max_depth, max_feature]), [n_est, max_depth, max_feature]
end

# use Hyperband for optimization
println("Hyperband")
hohb = @time @hyperopt for i=50,
        sampler=Hyperband(R=50, η=3, inner=RandomSampler(rng)),
        n_est = lower[1]:upper[1],
        max_depth = lower[2]:upper[2],
        max_feature = lower[3]:upper[3]
    if state !== nothing
        n_est, max_depth, max_feature = state
    end
    # println(i, "\t", n_est, "\t", max_depth, "\t", max_feature, "   \n")
    # res = Optim.optimize(x -> HPO_rf(round.(Int, x)), [n_est, max_depth, max_feature], float(lower), float(upper), NelderMead(), Optim.Options(f_calls_limit=round(Int, i)+1))
    # @show Optim.minimum(res), round.(Int, Optim.minimizer(res))
    HPO_rf([n_est, max_depth, max_feature]), [n_est, max_depth, max_feature]
end

# Hyperband with Bayesian optimization
println("Hyperband with Bayesian optimization")
hohbbo = @time @hyperopt for i=50,
        sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()], random_sampler=RandomSampler(rng))),
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

function plotHyperopt(ho, figname)
    params = hcat(ho.history...)'
    vals = [t[1] for t in ho.results]

    p1 = scatter(params[:, 1], params[:,2], zcolor=-vals, xlabel="n_estimators", ylabel="max_depth")
    plot!(p1, colorbar_title="Accuracy", colorbar = true, legend=false)
    p2 = scatter(params[:, 2], params[:,3], zcolor=-vals, xlabel="max_depth", ylabel="max_feature")
    plot!(p2,colorbar_title="Accuracy", colorbar = true, legend=false)
    p3 = scatter(params[:, 1], params[:,3], zcolor=-vals, xlabel="n_estimators", ylabel="max_feature")
    plot!(p3, colorbar_title="Accuracy", colorbar = true, legend=false)
    p_empty = plot(legend=false, ticks=nothing, border=:none)
    p = plot(p1, p_empty,p3, p2, size=(800, 600), layout=(2,2))
    savefig(p, figname)
end

function writeTestResult(dir, params_name, all_ho, all_cv)
    df_cv = DataFrame(all_cv)
    df_ho = DataFrame(hcat([ho.minimizer for ho in all_ho]...)', params_name)
    CSV.write(dir * "all_cv.csv", df_cv)
    CSV.write(dir * "all_ho_params.csv", df_ho)
end

plotHyperopt(hors, "../rf/rf_random_search.png")
plotHyperopt(hohb, "../rf/rf_hyperband.png")
plotHyperopt(hohbbo, "../rf/rf_hyperband_bo.png")

# use with test data and do cross validation
println("Cross validation")
println("Random Sampler")
println(hors.minimizer)
Random.seed!(42)
m1 = crossvalidate(rf(hors.minimizer), X_test_trans, Y_test, "accuracy_score", nfolds=5, verbose=true)
println("Hyperband with Random Sampler")
println(hohb.minimizer)
Random.seed!(42)
m2 = crossvalidate(rf(hohb.minimizer), X_test_trans, Y_test, "accuracy_score", nfolds=5, verbose=true)
println("Hyperband with BO")
println(hohbbo.minimizer)
Random.seed!(42)
m3 = crossvalidate(rf(hohbbo.minimizer), X_test_trans, Y_test, "accuracy_score", nfolds=5, verbose=true)
total_mean = [[t[1] for t in h.results] for h in [hors, hohb, hohbbo]]
# drop missing values
total_mean = [t[.!isnan.(t)] for t in total_mean]
p = boxplot(["Random Sampler"], -total_mean[1], ylabel="Accuracy", legend=false)
boxplot!(p, ["Hyperband with Random Sampler"], -total_mean[2])
boxplot!(p, ["Hyperband with BO"], -total_mean[3])
plot!(p, legend=false, ylabel="Accuracy score")
savefig(p, "../rf/rf_boxplot.png")

writeTestResult("./rf/", ["n_estimators", "max_depth", "max_feature"], [hors, hohb, hohbbo], [m1, m2, m3])

function writeToCSV(filename, ho, colnames)
    params = hcat(ho.history...)'
    vals = [t[1] for t in ho.results]
    CSV.write(filename, DataFrame(hcat(-vals, params), colnames))
end

writeToCSV("./rf/rf_random_search.csv", hors,
        ["accuracy", "n_estimators", "max_depth", "max_feature"])
writeToCSV("./rf/rf_hyperband_rs.csv", hohb,
        ["accuracy", "n_estimators", "max_depth", "max_feature"])
writeToCSV("./rf/rf_hyperband_bo.csv", hohbbo,
        ["accuracy", "n_estimators", "max_depth", "max_feature"])


# For Gradient Boosting
println("Gradient Boosting")
# Define hypterparameter function
HPO_gb = hp -> HPOLearner(gb(hp), X_train_trans, Y_train)
n_est_range = 10:500
lr_range = LinRange(0.01, 0.5, 100)
max_depth_range = 1:30


# random search
println("Random Search")
hors = @time @hyperopt for i=50,
        sampler = RandomSampler(rng), # This is default if none provided
        n_est = n_est_range,
        lr = lr_range,
        max_depth = max_depth_range
    # print(i, "\t", n_est, "\t", max_depth, "\t", max_feature, "   \t")
    params = [n_est, lr, max_depth]
    @show HPO_gb(params), params
end

# use Hyperband for optimization
println("Hyperband")
hohb = @time @hyperopt for i=50,
        sampler=Hyperband(R=50, η=3, inner=RandomSampler(rng)),
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
hohbbo = @time @hyperopt for i=50,
        sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()], random_sampler=RandomSampler(rng))),
        n_est = n_est_range,
        lr = lr_range,
        max_depth = max_depth_range
    if state !== nothing
        n_est, lr, max_depth = state
    end
    if !(n_est in n_est_range) || !(lr in lr_range) || !(max_depth in max_depth_range)
        # to prevent type error and circumvent bug in their code
        # the bug will step out of the search space
        @show NaN, [n_est, max_depth, max_feature]
    else
        @show HPO_gb([n_est, lr, max_depth]), [n_est, lr, max_depth]
    end
end

plotHyperopt(hors, "../gb/gb_random_search.png")
plotHyperopt(hohb, "../gb/gb_hyperband.png")
plotHyperopt(hohbbo, "../gb/gb_hyperband_bo.png")
# cross validation
println("Cross validation")
println("Random Sampler")
println(hors.minimizer)
Random.seed!(42)
m = crossvalidate(gb(hors.minimizer), X_test_trans, Y_test, "accuracy_score", nfolds=5, verbose=true)
println("Hyperband with Random Sampler")
println(hohb.minimizer)
Random.seed!(42)
crossvalidate(gb(hohb.minimizer), X_test_trans, Y_test, "accuracy_score", nfolds=5, verbose=true)
println("Hyperband with BO")
println(hohbbo.minimizer)
Random.seed!(42)
crossvalidate(gb(hohbbo.minimizer), X_test_trans, Y_test, "accuracy_score", nfolds=5, verbose=true)
