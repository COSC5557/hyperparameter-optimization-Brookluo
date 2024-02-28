# make sure local environment is activated
using Pkg
Pkg.activate(".")

using CSV
using Plots
using MLDataUtils

using Distributed
using DataFrames

# Add workers
nprocs() == 1 && addprocs(10, exeflags=["--project=$(Base.active_project())"])
workers()

@sync @everywhere using AutoMLPipeline
@sync @everywhere using DataFrames
@sync @everywhere using Random


# Load the data
df_red = CSV.read("winequality-red.csv", DataFrame)
# names = CSV.read("winequality.names", DataFrame)

# split the data into training and testing
# Random.seed!(42)
@everywhere rng = MersenneTwister(1234)
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
gb       = hp -> SKLearner("GradientBoostingClassifier", n_estimators=hp[1], learning_rate=hp[2], max_depth=[3], random_state=0)
svc      = hp -> SKLearner("SVC", C=hp[1], kernel=hp[2], degree=hp[3], random_state=0)
ada      = hp -> SKLearner("AdaBoostClassifier", n_estimators=hp[1], learning_rate=hp[2], random_state=0)
logistic = hp -> SKLearner("LogisticRegression", C=hp[1], random_state=0)


# Define the pipeline
@everywhere function HPOLearner(learner, X, Y)
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

    # use evolution algorithm to find the best hyperparameters for this model
    # we will use accuracy as the metric
    # we will use 5 fold cross validation
    Random.seed!(rng, 1234)
    # first transform the data
    # use OneHotEncoder for categorical data and RobustScaler for numerical data
    pl = @pipeline disc |> (catf |> ohe) + (numf |> rb |> pca) |> learner
    mean, sd, _ = crossvalidate(pl, X, Y, "accuracy_score", nfolds=5, verbose=false)
    return mean
end


# For Random Forest
println("Random Forest")
# Define hypterparameter function
@everywhere HPO_rf = hp -> HPOLearner(rf(round.(Int, hp)), X_train, Y_train)
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

@everywhere using Hyperopt

# random search
println("Random Search")
ho = @time @hyperopt for i=50,
        sampler = RandomSampler(rng), # This is default if none provided
        n_est = 10:300,
        max_depth = 1:30,
        max_feature = 1:30
    # print(i, "\t", n_est, "\t", max_depth, "\t", max_feature, "   \t")
    @show HPO_rf([n_est, max_depth, max_feature]), [n_est, max_depth, max_feature]
end
ho

# use Hyperband for optimization
println("Hyperband")
hohb = @time @phyperopt for i=50,
        sampler=Hyperband(R=50, η=3, inner=RandomSampler(rng)),
        n_est = 10:300,
        max_depth = 1:30,
        max_feature = 1:30
    if state !== nothing
        n_est, max_depth, max_feature = state
    end
    # println(i, "\t", n_est, "\t", max_depth, "\t", max_feature, "   \n")
    # res = Optim.optimize(HPO_rf, float([n_est, max_depth, max_feature]), float(lower), float(upper), NelderMead(), Optim.Options(f_calls_limit=round(Int, i)+1))
    # @show Optim.minimum(res), Optim.minimizer(res)
    # print(i, "\n")
    @show HPO_rf([n_est, max_depth, max_feature]), [n_est, max_depth, max_feature]
end
hohb

# Hyperband with Bayesian optimization
println("Hyperband with Bayesian optimization")
hohbbo = @time @phyperopt for i=50,
        sampler=Hyperband(R=50, η=3, inner=BOHB(dims=[Hyperopt.Continuous(), Hyperopt.Continuous(), Hyperopt.Continuous()], random_sampler=RandomSampler(rng))),
        n_est = 10:300,
        max_depth = 1:30,
        max_feature = 1:30
    if state !== nothing
        n_est, max_depth, max_feature = state
    end
    # print(i, "\n")
    @show HPO_rf([n_est, max_depth, max_feature]), [n_est, max_depth, max_feature]
end
hohbbo