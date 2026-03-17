"""
Evaluation metrics submodule.

Gini impurity scoring, downstream gene activity analysis, and related
evaluation utilities for communication programs and model predictions.
"""
module Eval

using Statistics
using LinearAlgebra
using StatsBase
using ..IO: scData, expression_matrix, var_table

include("gini.jl")
include("downstream_activity.jl")

export gini_impurity, gini_impurity_normalized
export downstream_gene_activity_score, get_avg_expression

end # module Eval
