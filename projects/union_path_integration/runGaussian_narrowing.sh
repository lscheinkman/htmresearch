#!/bin/bash

set -e -x

cd "$(dirname "$0")"
pwd

python convergence_simulation.py --bumpType gaussian --numModules 10 --numObjects 50 --numUniqueFeatures 40 --locationModuleWidth 10 --numSensations 9 --seed1 42 --seed2 42 --logCellActivity --resultName results/narrowing_gaussian.json

python plot_narrowing.py --inFile results/narrowing_gaussian.json --outFile1 narrowing_singleTrials_gaussian.pdf --outFile2 narrowing_aggregated_gaussian.pdf --exampleObjectCount 50 --exampleObjectNumbers 28 47 30 --scrambleCells --exampleModuleNumbers 0 2 9

python plot_rhombus_narrowing.py --inFile results/narrowing_gaussian.json --outFile narrowing_rhombus_gaussian.svg --objectNumber 46 --moduleNumbers 0 2 9 --numSteps 4


python convergence_simulation.py --bumpType gaussian --numModules 10 --numObjects 40 50 60 70 --numUniqueFeatures 40 --locationModuleWidth 10 --numSensations 9 --logCellActivity --resultName results/narrowing_aggregate_gaussian.json --repeat 10

python plot_aggregate_narrowing.py --inFile results/narrowing_aggregate_gaussian.json --outFile narrowing_aggregated_gaussian.pdf --objectCounts 40 50 60 70
