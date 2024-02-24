# make sure local environment is activated
using Pkg
Pkg.activate(".")

using CSV
using Plots
using MLDataUtils
using Random
using Evolutionary

using Distributed
using DataFrames

# Add workers
nprocs() == 1 && addprocs(10, exeflags=["--project=$(Base.active_project())"])
workers()

@sync @everywhere using AutoMLPipeline
@sync @everywhere using DataFrames


# Load the data
df_red = CSV.read("winequality-red.csv", DataFrame)
# names = CSV.read("winequality.names", DataFrame)

# split the data into training and testing
Random.seed!(42)
train, test = splitobs(shuffleobs(df_red), at = 0.7)
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
    # Random.seed!(42)
    # first transform the data
    # use OneHotEncoder for categorical data and RobustScaler for numerical data
    pl = @pipeline disc |> (catf |> ohe) + (numf |> rb |> pca) |> learner
    crossvalidate(pl, X, Y, "accuracy_score", nfolds=5, verbose=false)
end


# For Random Forest
println("Random Forest")
# Define hypterparameter function
HPO_rf = hp -> HPOLearner(rf(hp), X_train, Y_train)
# Cannot use EA here, because inputs are all integer. Use random search instead
# Random.seed!(42)
x0 = [100, 10, 10]
lower = [10, 1, 1]
upper = [200, 30, 30]
# random search
n_samp = 1000
params = hcat([rand(lower[i]:upper[i], n_samp) for i=1:3]...)
# tab = DataFrame(fetch.([Threads.@spawn HPO_rf(params[i, :]) for i=1:100]))
tab = @sync @distributed (vcat) for i=1:n_samp
    HPO_rf(params[i, :])
end
tab = DataFrame(tab)
# GA
# res = Evolutionary.optimize(HPO_rf, BoxConstraints(lower, upper), x0,
#                         GA(populationSize=100, crossoverRate=0.8, mutationRate=0.4),
#                         Evolutionary.Options(reltol=1e-4, iterations=10, show_trace=true, parallelization=:thread))

# For Gradient Boosting
# Define hypterparameter function
# HPO_gb = hp -> HPOLearner(gb(hp), X_train, Y_train)
# Cannot use EA here, because inputs are all integer. Use random search instead


# println(res)

# HPO_gb = hp -> HPOLearner(gb, fit_transform!(tran, X_train, Y_train), Y_train)
# HPO_svc = hp -> HPOLearner(svc, fit_transform!(tran, X_train, Y_train), Y_train)
# HPO_mlp = hp -> HPOLearner(mlp, fit_transform!(tran, X_train, Y_train), Y_train)