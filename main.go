package main

//
// Use a genetic algorithm to solve the 8 queens problem, where 8 queens must be placed on a chess board without attacking each other.
//

import (
    "os"
    "fmt"
    "sync"
    "time"
    "math/rand"
    "runtime"
    "math"
    "sort"
    "strings"
)

const BOARD_SIZE = 8           // board width and height
const BOARD_SPACES = BOARD_SIZE * BOARD_SIZE

type NQueensProblem struct {}

func (prob NQueensProblem) MeasureFitness(genome []byte) int {
    return rateBoard(string(genome))
}

func (prob NQueensProblem) AllValidBytes() []byte {
    return []byte("Qx")
}

func (prob NQueensProblem) GenomeLength() int {
    return BOARD_SPACES    // every gene in the genome represents one space on the board
}

//
// Implement this interface to use Solve(), which will atempt to solve the problem using a genetic algorithm
//
type GAProblem interface {
    MeasureFitness(genome []byte) int
    AllValidBytes() []byte
    GenomeLength() int
}

//
// A collection of individuals
//
type Population struct {
    Individuals []Individual
    GenerationCount int       // number of generations completed by this population
    HighestFitness int        // highest fitness in the population
    FitnessStagnation int     // number of most recent generations in which the highest fitness has not improved
}

//
// A member of the population, representing a possible solution
//
type Individual struct {
    Genome []byte
    Fitness int          // fitness rating, measured by heuristic function (starts at -1)
    AncestorCount int    // how many predecessor individuals led to this one, starting at 0
}

//
// All the configuration needed to run a genetic algorithm
//
//     firstPopulations -- list of Populations to start with; set to nil to have new populations generated from scratch
//     maxGenerations   -- stop after this many generations have been made (default: 300)
//     maxDuration      -- stop after this amount of time has passed (default: 10 minutes)
//     populationSize   -- number of individuals in each population (default: 300)
//     toKill           -- how many members of the population to kill each generation (default: 150)
//     immuneCount      -- how many of the most fit are immune from being killed each generation (default: 5)
//     transferRate     -- how many generations to wait in fitness stagnation before transfering some Individuals between Populations (default: 5)
//     mutationPolicy   -- specifies how mutation probability changes over time
//     goroutineCount   -- the number of goroutines to use
//
type GAConfig struct {
    Problem GAProblem
    FirstPopulations []Population
    MaxGenerations int
    MaxDuration time.Duration
    PopulationSize int
    ToKill int
    ImmuneCount int
    TransferRate int
    MutationPolicy MutationPolicy
    GoroutineCount int
}

//
// The probability out of 1,000 that a byte in a Gene will mutate.
// The equation is: m = y(fitnessStagnation) + z
// y and z are all arbitrarily chosen floating point numbers between -3 and 3, inclusive.
//
type MutationPolicy struct {
    Y float32 // Y and Z are used in a linear equation that makes mutation probability correlate with generation count
    Z float32
}

// Reasonable configuration settings for a genetic algorithm
var defaultConfig GAConfig = GAConfig {
    MaxGenerations: math.MaxInt32,
    MaxDuration: time.Second * time.Duration(10),
    PopulationSize: 300,
    ToKill: 150,          // how many members of the population to kill each generation
    ImmuneCount: 5,       // how many of the most fit are immune from being killed each generation
    TransferRate: 5,      // after this many generations have passed, and the most fit individual in a population still
                          // isn't any more fit, pull and push to the shared channel to get outside help from other populations
    MutationPolicy: MutationPolicy{Y: 0.0015, Z: 30.0},
    GoroutineCount: runtime.NumCPU() - 1}

//
// Return an integer from 0 to 1000 that measures how fit a board possition is, with higher numbers being more fit
//
func rateBoard(board string) int {
    rating := 0

    // examine all columns
    for x:=0; x<BOARD_SIZE; x++ {

        // check all spaces in this column for queens
        queenCount := 0
        for i:=x; i<BOARD_SPACES; i+=BOARD_SIZE {
            if string(board[i]) == "Q" {
                queenCount += 1
            }
        }

        // reduce rating if there are multiple queens
        if queenCount > 1 {
            rating -= queenCount
        }
    }

    // examine all rows
    for y:=0; y<BOARD_SIZE; y++ {

        // check all spaces in this row for queens
        queenCount := 0
        for i:=y*BOARD_SIZE; i<(y*BOARD_SIZE)+BOARD_SIZE; i++ {
            if string(board[i]) == "Q" {
                queenCount += 1
            }
        }

        // reduce rating if there are multiple queens
        if queenCount > 1 {
            rating -= queenCount
        }
    }

    // examine all forward-slanted diagonals on the top half of the board
    //      checks these array indexes on an 8x8 board (each row is one iteration):
    //      0
    //      8,  1
    //      16, 9,  2
    //      24, 17, 10, 3
    //      ...
    for y:=0; y<BOARD_SPACES; y+=BOARD_SIZE {

        // check all spaces on this diagonal for queens
        queenCount := 0
        for i:=y; i>=0; i-=BOARD_SIZE-1 {
            if string(board[i]) == "Q" {
                queenCount += 1
            }
        }

        // reduce rating if there are multiple queens
        if queenCount > 1 {
            rating -= queenCount
        }
    }

    // examine all forward-slanted diagonals on the bottom half of the board
    //      checks these array indexes on an 8x8 board (each row is one iteration):
    //      63
    //      62, 55
    //      61, 54, 47
    //      60, 53, 46, 39
    //      ...
    for y:=BOARD_SPACES-1; y>=0; y-=BOARD_SIZE {

        // check all spaces on this diagonal for queens
        queenCount := 0
        for i:=y; i<BOARD_SPACES; i+=BOARD_SIZE-1 {
            if string(board[i]) == "Q" {
                queenCount += 1
            }
        }

        // reduce rating if there are multiple queens
        if queenCount > 1 {
            rating -= queenCount
        }
    }

    // examine all backward-slanted diagonals on the top half of the board
    //      checks these array indexes on an 8x8 board:
    //      7
    //      6, 15
    //      5, 14, 23
    //      4, 13, 22, 31
    //      ...
    for y:=BOARD_SIZE-1; y<BOARD_SPACES; y+=BOARD_SIZE {

        // check all spaces on this diagonal for queens
        queenCount := 0
        for i:=y; i>=0; i-=BOARD_SIZE+1 {
            if string(board[i]) == "Q" {
                queenCount += 1
            }
        }

        // reduce rating if there are multiple queens
        if queenCount > 1 {
            rating -= queenCount
        }
    }

    // examine all backward-slanted diagonals on the bottom half of the board
    //      checks these array indexes on an 8x8 board:
    //      56
    //      57, 48
    //      58, 49, 40
    //      59, 50, 41, 32
    //      ...
    for y:=BOARD_SPACES-BOARD_SIZE; y>=0; y-=BOARD_SIZE {

        // check all spaces on this diagonal for queens
        queenCount := 0
        for i:=y; i<BOARD_SPACES; i+=BOARD_SIZE+1 {
            if string(board[i]) == "Q" {
                queenCount += 1
            }
        }

        // reduce rating if there are multiple queens
        if queenCount > 1 {
            rating -= queenCount
        }
    }

    // reduce fitness if the number of queens on the board is not BOARD_SIZE (8 on an 8x8 board)
    queenCount := strings.Count(board, "Q")
    abs := queenCount - BOARD_SIZE
    if abs < 0 {
        abs *= -1
    }
    rating -= abs * 50   // -50 points for each extra or each missing queen

    return rating
}

//
// Wrapper for SolveWithConfig() that uses sensible defaults
//
func Solve(problem GAProblem) ([]Population, error) {
    defaultConfig.Problem = problem
    return SolveWithConfig(defaultConfig)
}

//
// Using custom configuration values, solve a problem using an evolutionary algorithm strategy to do so.
// Return the final populations after a stopping point has been reached.
//
func SolveWithConfig(config GAConfig) ([]Population, error) {

    // Error out if the configuration settings are not valid
    err := validateGAConfig(config)
    if err != nil {
        return []Population{}, err
    }

    // Make a random number generator
    generator := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))

    // Start the timer that doesn't let this function run past the maximum allowed time
    haltTimer := time.NewTimer(config.MaxDuration)

    // Add an entry to FirstPopulations if the list is empty (having zero populations to work with is not valid)
    if len(config.FirstPopulations) == 0 {
        config.FirstPopulations = append(config.FirstPopulations, MakePopulation(config.Problem, config.PopulationSize, config.GoroutineCount))
    }

    // Make a list of populations to operate on that is the same size as the list of FirstPopulations
    // Where FirstPopulations contains a population with an empty list of Individuals, add Individuals
    var populations []Population
    for i:=0; i<len(config.FirstPopulations); i++ {
        if len(config.FirstPopulations[i].Individuals) == 0 {
            populations = append(populations, MakePopulation(config.Problem, config.PopulationSize, config.GoroutineCount))
        } else {
            populations = append(populations, config.FirstPopulations[i])
        }
    }

    // Cycle through every population, considering each one separately, one at a time
    halt := false
    for halt == false {
        for p:=0; p<len(populations); p++ {

            // Sort by fitness in reverse order, with most fit first
            sort.Slice(populations[p].Individuals, func (i, j int) bool { return populations[p].Individuals[i].Fitness > populations[p].Individuals[j].Fitness })

            // If there is a stagnation problem, transfer the most fit individual from another population into the current one
            if populations[p].FitnessStagnation >= config.TransferRate {
                mostFit := FindMostFitIndividual(populations)
                if mostFit.Fitness > populations[p].HighestFitness {
                    populations[p].Individuals[len(populations[p].Individuals)-1] = mostFit
                }
            }

            kill(&populations[p], config.ToKill, config.ImmuneCount, generator)

            mutationRate := calculateMutationRate(config.MutationPolicy, populations[p].FitnessStagnation)
            repopulate(config.Problem, &populations[p], config.ToKill, mutationRate, config.GoroutineCount)

            // Update highest fitness
            newHighest := findHighestFitness(populations[p])
            if newHighest > populations[p].HighestFitness {
                populations[p].HighestFitness = newHighest
                populations[p].FitnessStagnation = 0
            } else {
                populations[p].FitnessStagnation++
            }

            // Halt if the max number of generations has been reached
            populations[p].GenerationCount += 1
            if populations[p].GenerationCount >= config.MaxGenerations {
                halt = true
                break
            }

            // Halt if time has exceeded config.MaxDuration
            select {
            case <-haltTimer.C:
                halt = true
                break
            default:
            }
        }
    }

    return populations, nil
}

//
// Print summary information about a list of Populations
//
func SummarizePopulations(populations []Population) {
    fmt.Printf("\n%40s\n", "=== Summary of Populations ===")
    fmt.Printf("%-42s %d\n", "Number of populations:", len(populations))

    var fitnesses []int
    for _,pop := range populations {
        fitnesses = append(fitnesses, pop.Individuals[0].Fitness)
    }
    sort.Sort(sort.Reverse(sort.IntSlice(fitnesses)))
    fmt.Printf("%-42s %v\n", "Fitness, high to low:", fitnesses)

    ancestorCount := -1
    for _,pop := range populations {
        for _,individual := range pop.Individuals {
            if individual.Fitness == fitnesses[0] {
                ancestorCount = individual.AncestorCount
            }
        }
    }
    fmt.Printf("%-42s %d\n", "Ancestor count of the most fit individual:", ancestorCount)
    fmt.Printf("%-42s %d\n", "Generation count of the first population:", populations[0].GenerationCount)
    fmt.Printf("%-42s %d\n", "Ending stagnation of the first population:", populations[0].FitnessStagnation)
    fmt.Printf("%-42s %.2f%%\n", "Efficiency:", (float64(ancestorCount)/float64(populations[0].GenerationCount)) * 100)
    fmt.Println()
}

//
// Return the most fit Individual among all populations
//
func FindMostFitIndividual(populations []Population) Individual {
    var bestIndividual Individual
    bestFitness := math.MinInt32
    for _,pop := range populations {
        for _,individual := range pop.Individuals {
            if individual.Fitness > bestFitness {
                bestFitness = individual.Fitness
                bestIndividual = individual
            }
        }
    }
    return bestIndividual
}

//
// Make a starting population of individuals from scratch
//
func MakePopulation(problem GAProblem, size int, goroutineCount int) Population {
    var population Population
    population.Individuals = makeIndividuals(problem, nil, size, -1, goroutineCount)
    population.GenerationCount = 0  // number of generations completed by this population
    population.FitnessStagnation = 0 // number of most recent generations in which the highest fitness has not improved
    population.HighestFitness = math.MinInt32        // highest fitness in the population
    return population
}

//
// Delete some of the less fit members
// population must already be sorted by fitness with most fit at the beginning
// k is the number to kill
// immune specifies the number of most fit members of the population that won't be killed (2 means the two most fit are immune)
//
func kill(population *Population, k int, immune int, generator *rand.Rand) {
    for i:=0; i<k; i++ {

        // find the highest extisting fitness value of all individuals
        maxFitness := findHighestFitness(*population)

        // build a list of integers that track how far off the corresponding individual was from the max fitness
        var shortOfMax []int
        totalShort := 0       // the total difference from max for all individuals
        for n:=0; n<len(population.Individuals); n++ {
            short := maxFitness - population.Individuals[n].Fitness
            totalShort += short
            shortOfMax = append(shortOfMax, short)
        }

        // if the total is 0, then kill anything; there's no reason to choose one over another
        if totalShort == 0 {
            deleteFromPop(population, generator.Intn(len(population.Individuals) - immune) + immune)
        } else {
            inc := 0
            r := generator.Intn(totalShort)
            for n:=immune; n<len(population.Individuals); n++ {
                if inc + shortOfMax[n] > r {
                    deleteFromPop(population, n)
                    break
                }
                inc += shortOfMax[n]
            }
        }

    }
}

//
// Return a whole number that specifies how many times out of 1,000 any given byte in a Genome will mutate
// The expression is: y(fitnessStagnation) + z
// fitnessStagnation is the number of generations that have passed with no increased fitness for the most fit individual in the population.
// y and z are all arbitrarily chosen numbers between -3 and 3, inclusive.
//
func calculateMutationRate(policy MutationPolicy, fitnessStagnation int) int {
    return int(policy.Y * float32(fitnessStagnation) + policy.Z)
}

//
// The opposite of kill -- uses generateOffspring() to add new members to the population
// toAdd is the number of new members to add
//
func repopulate(problem GAProblem, population *Population, toAdd int, mutationRate int, goroutineCount int) {
    newIndividuals := makeIndividuals(problem, population, toAdd, mutationRate, goroutineCount)
    population.Individuals = append(population.Individuals, newIndividuals...)
}

//
// Find the maximum fitness value among the given individuals
//
func findHighestFitness(population Population) int {
    maxFitness := math.MinInt32
    for i:=0; i<len(population.Individuals); i++ {
        if population.Individuals[i].Fitness > maxFitness {
            maxFitness = population.Individuals[i].Fitness
        }
    }
    return maxFitness
}

//
// Use goroutineCount number of goroutines to make toMake number of Individuals
// If population is not nil, then new Individuals will be based on existing Individuals in that population
// mutationRate is ignored if population is nil
//
func makeIndividuals(problem GAProblem, population *Population, toMake int, mutationRate int, goroutineCount int) []Individual {

    // Ensure a reasonable goroutineCount
    if goroutineCount < 1 { goroutineCount = 1 }

    // Build up a list of new Individuals to return as the result
    var newIndividuals []Individual
    for i:=0; i<toMake; i++ {
        newIndividuals = append(newIndividuals, Individual{})
    }

    // Populate newIndividuals using goroutines running in parallel
    var wg sync.WaitGroup
    assignments := assignGoroutines(toMake, goroutineCount)
    for i:=0; i<goroutineCount; i++ {
        wg.Add(1)
        go func(indexesToChange []int) {
            defer wg.Done()
            generator := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
            for _,index := range(indexesToChange) {
                if population == nil {
                    var newGenome []byte
                    for n:=0; n<problem.GenomeLength(); n++ {
                        newGenome = append(newGenome, problem.AllValidBytes()[generator.Intn(len(problem.AllValidBytes()))])
                    }

                    newIndividuals[index] = Individual{Fitness: problem.MeasureFitness(newGenome), Genome: newGenome, AncestorCount: 0}
                } else {
                    parent1, parent2 := selectTwo(*population, generator)
                    newIndividuals[index] = generateOffspring(problem, population.Individuals[parent1], population.Individuals[parent2], mutationRate, generator)
                }
            }
        }(assignments[i])
    }
    wg.Wait()

    return newIndividuals
}

//
// Remove individual at the specified index
//
func deleteFromPop(population *Population, index int) {
    population.Individuals = append(population.Individuals[:index], population.Individuals[index+1:]...)
}

//
// Produce an error if any configuration settings are not valid, return nil otherwise
//
func validateGAConfig(config GAConfig) error {

    if len(config.Problem.AllValidBytes()) == 0 {
        return fmt.Errorf("AllValidBytes() cannot have length 0.")
    }
    if config.Problem.GenomeLength() <= 0 {
        return fmt.Errorf("GenomeLength() must return a value greater than 0. The value returned was %d.", config.Problem.GenomeLength())
    }
    if len(config.FirstPopulations) < 0 {
        return fmt.Errorf("FirstPopulations cannot have negative length.")
    }
    if config.MaxGenerations <= 0 {
        return fmt.Errorf("MaxGenerations must be greater than 0. The value given was %d.", config.MaxGenerations)
    }
    if config.MaxDuration <= 0 {
        return fmt.Errorf("MaxDuration must be greater than 0. The value given was %v.", config.MaxDuration)
    }
    if config.PopulationSize <= 0 {
        return fmt.Errorf("PopulationSize must be greater than 0. The value given was %d.", config.PopulationSize)
    }
    if config.ToKill <= 0 {
        return fmt.Errorf("ToKill must be greater than 0. The value given was %d.", config.ToKill)
    }
    if config.ImmuneCount < 0 {
        return fmt.Errorf("ImmuneCount must be 0 or greater. The value given was %d.", config.ImmuneCount)
    }
    if config.TransferRate <= 0 {
        return fmt.Errorf("TransferRate must be greater than 0. The value given was %d.", config.TransferRate)
    }
    if config.GoroutineCount <= 0 {
        return fmt.Errorf("GoroutineCount must be greater than 0. The value given was %d.", config.GoroutineCount)
    }
    if config.ToKill > config.PopulationSize {
        return fmt.Errorf("ToKill must not be greater than PopulationSize. ToKill is %d; PopulationSize is %d.", config.ToKill, config.PopulationSize)
    }
    if config.ImmuneCount > config.PopulationSize {
        return fmt.Errorf("ImmuneCount must not be greater than PopulationSize. ImmuneCount is %d; PopulationSize is %d.", config.ImmuneCount, config.PopulationSize)
    }
    if config.ToKill + config.ImmuneCount > config.PopulationSize {
        return fmt.Errorf("ToKill + ImmuneCount must not be greater than PopulationSize. ToKill is %d; ImmuneCount is %d; PopulationSize is %d.", config.ToKill, config.ImmuneCount, config.PopulationSize)
    }

    return nil
}

//
// Make a list, with each item being a list of slots (usually indexes of Individuals) that a particular goroutine should be assigned to work on. Each slot can be assigned to only one goroutine.
// This is used to spread out the work that multiple goroutines are doing in parallel to create new Individuals.
//
func assignGoroutines(slotsRemaining int, goroutineCount int) [][]int {

    // Make a list as long as the number of goroutines to store the results
    assignments := [][]int{}
    for i:=0; i<goroutineCount; i++ {
        assignments = append(assignments, []int{})
    }

    // Solve this problem recursively
    return assignGoroutines_(slotsRemaining, assignments)
}

func assignGoroutines_(slotsRemaining int, assignments [][]int) [][]int {

    // No goroutines to assign slots to
    if len(assignments) == 0 {
        return assignments
    }

    // No slots to assign goroutines to
    if slotsRemaining <= 0 {
        return assignments
    }

    // Decide what slots the next goroutine should be assigned to
    slotCount := slotsRemaining / len(assignments)
    for i:=0; i<slotCount; i++ {
        slotsRemaining--
        assignments[0] = append(assignments[0], slotsRemaining)
    }
    return append([][]int{assignments[0]}, assignGoroutines_(slotsRemaining, assignments[1:])...)
}

//
// Return two unique indexes, each representing a weighted random member of the population
//
func selectTwo(population Population, generator *rand.Rand) (int, int) {
    if len(population.Individuals) < 2 {
        panic("Error: Cannot select 2 from a population size less than 2")
    }

    if len(population.Individuals) == 2 {
        return 0, 1
    }

    index1 := weightedSelect(population, generator)
    index2 := weightedSelect(population, generator)

    for index1 == index2 {
        index2 = weightedSelect(population, generator)
    }

    return index1, index2
}

//
// Return the index for a random member of the population, with fitness increasing the likelihood of selection
//
func weightedSelect(population Population, generator *rand.Rand) int {

    // calculate a number to add to all fitness values that makes them positive
    adjustment := 0
    minFitness := findMinFitness(population)
    if minFitness < 0 {
        adjustment = -minFitness + 1  // +1 because totalFitness can't end up being 0
    }

    // find the total fitness among all members
    totalFitness := 0
    for i:=0; i<len(population.Individuals); i++ {
        totalFitness += population.Individuals[i].Fitness + adjustment
    }

    inc := 0
    r := generator.Intn(totalFitness)
    for i:=0; i<len(population.Individuals); i++ {
        if inc + population.Individuals[i].Fitness + adjustment > r {
            return i
        }
        inc += population.Individuals[i].Fitness + adjustment
    }

    // should not get this far
    panic("Unexpected event in weightedSelect()")
    return -1
}

//
// Generate one offspring from two parent individuals
//
func generateOffspring(problem GAProblem, parent1 Individual, parent2 Individual, mutationRate int, generator *rand.Rand) Individual {

    if len(parent1.Genome) != len(parent2.Genome) {
        panic("Error: Not equal lengths")
    }

    var result []byte
    if generator.Intn(100) > 2 {
        for i:=0; i<len(parent1.Genome); i++ {
            if flipCoin(generator) {
                result = append(result, parent1.Genome[i])
            } else {
                result = append(result, parent2.Genome[i])
            }
        }
    } else {
        if flipCoin(generator) {
            result = append(result, parent1.Genome...)
        } else {
            result = append(result, parent2.Genome...)
        }
    }

    result = mutate(problem, result, mutationRate, generator)
    return Individual{Fitness: problem.MeasureFitness(result), Genome: result, AncestorCount: incrementAncestorCount(parent1, parent2)}
}

//
// Find the minimum fitness value among the given individuals
//
func findMinFitness(population Population) int {
    minFitness := population.Individuals[0].Fitness
    for i:=1; i<len(population.Individuals); i++ {
        if population.Individuals[i].Fitness < minFitness {
            minFitness = population.Individuals[i].Fitness
        }
    }
    return minFitness
}

//
// Return true or false with a 50/50 chance
//
func flipCoin(generator *rand.Rand) bool {
    return generator.Intn(2) == 0
}

//
// Return a copy of a genome that may be mutated
//
func mutate(problem GAProblem, genome []byte, mutationRate int, generator *rand.Rand) []byte {
    var result []byte
    for i:=0; i<len(genome); i++ {
        if generator.Intn(1000) < mutationRate {
            result = append(result, problem.AllValidBytes()[generator.Intn(len(problem.AllValidBytes()))])  // pick a random new byte
        } else {
            result = append(result, genome[i])     // keep the existing byte, unmutated
        }
    }
    return result
}

//
// Return the greater ancestor count between two parent individuals plus one
//
func incrementAncestorCount(parent1 Individual, parent2 Individual) int {
    if parent1.AncestorCount > parent2.AncestorCount {
        return parent1.AncestorCount + 1
    } else {
        return parent2.AncestorCount + 1
    }
}

//
// Program execution begins here
//
func main() {
    var finalPopulations []Population
    var err error
    finalPopulations, err = Solve(NQueensProblem{})
    if err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }

    SummarizePopulations(finalPopulations)
    bestIndividual := FindMostFitIndividual(finalPopulations)
    fmt.Println(string(bestIndividual.Genome))
}

