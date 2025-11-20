package viterbi

import (
	"fmt"
	"math"
)

type State interface {
	ID() int
}
type Observation interface {
	ID() int
}

type TransitionHash struct {
	From State
	To   State
}

type EmissionHash struct {
	State       State
	Observation Observation
}

type Viterbi struct {
	states                  []State
	observations            []Observation
	startProbabilities      map[State]float64
	emissionProbabilities   map[EmissionHash]float64
	transitionProbabilities map[TransitionHash]float64
}

type ViterbiPath struct {
	Probability float64
	Path        []State
}

type ViterbiVal struct {
	prob float64
	prev State
}

func New() *Viterbi {
	return &Viterbi{
		startProbabilities:      make(map[State]float64),
		emissionProbabilities:   make(map[EmissionHash]float64),
		transitionProbabilities: make(map[TransitionHash]float64),
	}
}

func (v *Viterbi) AddState(s State) {
	v.states = append(v.states, s)
}

func (v *Viterbi) AddObservation(obs Observation) {
	v.observations = append(v.observations, obs)
}

func (v *Viterbi) PutStartProbability(state State, val float64) {
	v.startProbabilities[state] = val
}

func (v *Viterbi) PutEmissionProbability(s State, obs Observation, val float64) {
	emKey := EmissionHash{s, obs}
	v.emissionProbabilities[emKey] = val
}

func (v *Viterbi) PutTransitionProbability(f State, t State, val float64) {
	trKey := TransitionHash{f, t}
	v.transitionProbabilities[trKey] = val
}

// EvalPath see ref bellow
// https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
// When every probability is in [0;1]
func (v Viterbi) EvalPath() ViterbiPath {
	// Check for edge cases
	if len(v.observations) == 0 {
		return ViterbiPath{Probability: 0.0, Path: nil}
	}
	if len(v.states) == 0 {
		return ViterbiPath{Probability: 0.0, Path: nil}
	}

	var (
		V     = make([]map[State]ViterbiVal, len(v.observations))
		path  map[State][]State
		maxPr = 0.0
	)

	// Initialize first time step
	V[0] = make(map[State]ViterbiVal)
	path = make(map[State][]State)

	for s := range v.states {
		st := v.states[s]
		if _, ok := v.startProbabilities[st]; !ok {
			continue
		}
		V[0][st] = ViterbiVal{prob: v.startProbabilities[st] * v.emissionProbabilities[EmissionHash{st, v.observations[0]}]}
		path[st] = append(path[st], st)
	}

	for t := 1; t < len(v.observations); t++ {
		V[t] = make(map[State]ViterbiVal)
		for s1 := range v.states {
			s := v.states[s1]
			if _, ok := v.emissionProbabilities[EmissionHash{s, v.observations[t]}]; !ok {
				// No emission for current state of current observation
				continue
			}
			maxTransitionProbability := 0.0
			var tmpState State
			foundValidTransition := false
			for s2 := range v.states {
				r := v.states[s2]
				vTransition, ok := v.transitionProbabilities[TransitionHash{r, s}]
				if !ok {
					// No transition between states
					continue
				}
				stateProb, ok := V[t-1][r]
				if !ok {
					// No probability from state to observation
					continue
				}
				transitionProbability := vTransition * stateProb.prob
				if !foundValidTransition || transitionProbability > maxTransitionProbability {
					maxTransitionProbability = transitionProbability
					tmpState = r
					foundValidTransition = true
				}
			}
			if !foundValidTransition {
				// No valid transition found for this state
				continue
			}
			maxProbability := maxTransitionProbability * v.emissionProbabilities[EmissionHash{s, v.observations[t]}]
			V[t][s] = ViterbiVal{prob: maxProbability, prev: tmpState}
		}
	}

	// Find the maximum probability in the last time step
	if len(V[len(V)-1]) == 0 {
		// No valid path found
		return ViterbiPath{Probability: maxPr, Path: nil}
	}

	for _, value := range V[len(V)-1] {
		if value.prob > maxPr {
			maxPr = value.prob
		}
	}

	opt := []State{}
	var previous State
	foundFinalState := false
	for st, value := range V[len(V)-1] {
		if value.prob == maxPr {
			opt = append(opt, st)
			previous = st
			foundFinalState = true
			break
		}
	}

	if !foundFinalState {
		return ViterbiPath{Probability: maxPr, Path: nil}
	}

	// Backtrack to find the path
	for t := len(V) - 2; t >= 0; t-- {
		opt = append([]State{V[t+1][previous].prev}, opt...)
		previous = V[t+1][previous].prev
	}

	return ViterbiPath{maxPr, opt}
}

// EvalPathLogProbabilities When every probability is logarithmic
func (v Viterbi) EvalPathLogProbabilities() ViterbiPath {
	// Check for edge cases
	if len(v.observations) == 0 {
		return ViterbiPath{Probability: math.Inf(-1), Path: nil}
	}
	if len(v.states) == 0 {
		return ViterbiPath{Probability: math.Inf(-1), Path: nil}
	}

	var (
		V     = make([]map[State]ViterbiVal, len(v.observations))
		path  map[State][]State
		maxPr = math.Inf(-1)
	)

	// Initialize first time step
	V[0] = make(map[State]ViterbiVal)
	path = make(map[State][]State)

	for s := range v.states {
		st := v.states[s]
		if _, ok := v.startProbabilities[st]; !ok {
			continue
		}
		V[0][st] = ViterbiVal{prob: v.startProbabilities[st] + v.emissionProbabilities[EmissionHash{st, v.observations[0]}]}
		path[st] = append(path[st], st)
	}

	for t := 1; t < len(v.observations); t++ {
		V[t] = make(map[State]ViterbiVal)
		for s1 := range v.states {
			s := v.states[s1]
			if _, ok := v.emissionProbabilities[EmissionHash{s, v.observations[t]}]; !ok {
				// No emission for current state of current observation
				continue
			}
			maxTransitionProbability := math.Inf(-1)
			var tmpState State
			foundValidTransition := false
			for s2 := range v.states {
				r := v.states[s2]
				vTransition, ok := v.transitionProbabilities[TransitionHash{r, s}]
				if !ok {
					// No transition between states
					continue
				}
				stateProb, ok := V[t-1][r]
				if !ok {
					// No probability from state to observation
					continue
				}
				transitionProbability := vTransition + stateProb.prob
				if !foundValidTransition || transitionProbability > maxTransitionProbability {
					maxTransitionProbability = transitionProbability
					tmpState = r
					foundValidTransition = true
				}
			}
			if !foundValidTransition {
				// No valid transition found for this state
				continue
			}
			maxProbability := maxTransitionProbability + v.emissionProbabilities[EmissionHash{s, v.observations[t]}]
			V[t][s] = ViterbiVal{prob: maxProbability, prev: tmpState}
		}
	}

	// Find the maximum probability in the last time step
	if len(V[len(V)-1]) == 0 {
		// No valid path found
		return ViterbiPath{Probability: maxPr, Path: nil}
	}

	for _, value := range V[len(V)-1] {
		if value.prob > maxPr {
			maxPr = value.prob
		}
	}

	opt := []State{}
	var previous State
	foundFinalState := false
	for st, value := range V[len(V)-1] {
		if value.prob == maxPr {
			opt = append(opt, st)
			previous = st
			foundFinalState = true
			break
		}
	}

	if !foundFinalState {
		return ViterbiPath{Probability: maxPr, Path: nil}
	}

	// Backtrack to find the path
	for t := len(V) - 2; t >= 0; t-- {
		opt = append([]State{V[t+1][previous].prev}, opt...)
		previous = V[t+1][previous].prev
	}

	return ViterbiPath{maxPr, opt}
}

func printPathTable(V []map[State]ViterbiVal) {
	fmt.Printf("    ")
	for i := 0; i < len(V); i++ {
		fmt.Printf("%7d ", i)
	}
	fmt.Println("")

	for y := range V[0] {
		fmt.Printf("%.5s:  ", y)
		for t := 0; t < len(V); t++ {
			fmt.Printf("%.5f ", V[t][y].prob)
		}
		fmt.Println("")
	}
}
