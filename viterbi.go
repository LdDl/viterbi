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
	observation Observation
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
	if v.startProbabilities == nil {
		v.startProbabilities = make(map[State]float64)
	}
	v.startProbabilities[state] = val
}

func (v *Viterbi) PutEmissionProbability(s State, obs Observation, val float64) {
	if v.emissionProbabilities == nil {
		v.emissionProbabilities = make(map[EmissionHash]float64)
	}
	emKey := EmissionHash{s, obs}
	if _, ok := v.emissionProbabilities[emKey]; !ok {
		v.emissionProbabilities[emKey] = val
	}
}

func (v *Viterbi) PutTransitionProbability(f State, t State, val float64) {
	if v.transitionProbabilities == nil {
		v.transitionProbabilities = make(map[TransitionHash]float64)
	}
	trKey := TransitionHash{f, t}
	if _, ok := v.transitionProbabilities[trKey]; !ok {
		v.transitionProbabilities[trKey] = val
	}
}

// EvalPath see ref bellow
// https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
// When every probability is in [0;1]
func (v Viterbi) EvalPath() ViterbiPath {
	var (
		V     []map[State]ViterbiVal
		path  map[State][]State
		maxPr = -math.MaxFloat64
	)

	V = append(V, make(map[State]ViterbiVal))
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
		V = append(V, make(map[State]ViterbiVal))
		for s1 := range v.states {
			s := v.states[s1]
			if _, ok := v.emissionProbabilities[EmissionHash{s, v.observations[t]}]; !ok {
				// No emission for current state of current observation
				continue
			}
			maxTransitionProbability := -math.MaxFloat64
			tmpState := v.states[0]
			for s2 := range v.states {
				r := v.states[s2]
				if _, ok := v.transitionProbabilities[TransitionHash{r, s}]; !ok {
					// No transition between states
					continue
				}
				if _, ok := V[t-1][r]; !ok {
					// No probability from state to observation
					continue
				}
				transitionProbability := V[t-1][r].prob * v.transitionProbabilities[TransitionHash{r, s}]
				if transitionProbability > maxTransitionProbability {
					maxTransitionProbability = transitionProbability
					tmpState = r
				}
			}
			maxProbability := maxTransitionProbability * v.emissionProbabilities[EmissionHash{s, v.observations[t]}]
			V[t][s] = ViterbiVal{prob: maxProbability, prev: tmpState}
		}
	}

	for _, value := range V[len(V)-1] {
		if value.prob > maxPr {
			maxPr = value.prob
		}
	}

	opt := []State{}
	var previous State
	for st, value := range V[len(V)-1] {
		if value.prob == maxPr {
			opt = append(opt, st)
			previous = st
			break
		}
	}
	for t := len(V) - 2; t >= 0; t-- {
		opt = append([]State{V[t+1][previous].prev}, opt...)
		previous = V[t+1][previous].prev
	}

	return ViterbiPath{maxPr, opt}
}

// EvalPathLogProbabilities When every probability is logarithmic
func (v Viterbi) EvalPathLogProbabilities() ViterbiPath {
	var (
		V     []map[State]ViterbiVal
		path  map[State][]State
		maxPr = -math.MaxFloat64
	)

	V = append(V, make(map[State]ViterbiVal))
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
		V = append(V, make(map[State]ViterbiVal))
		for s1 := range v.states {
			s := v.states[s1]
			if _, ok := v.emissionProbabilities[EmissionHash{s, v.observations[t]}]; !ok {
				// No emission for current state of current observation
				continue
			}
			maxTransitionProbability := -math.MaxFloat64
			tmpState := v.states[0]
			for s2 := range v.states {
				r := v.states[s2]
				if _, ok := v.transitionProbabilities[TransitionHash{r, s}]; !ok {
					// No transition between states
					continue
				}
				if _, ok := V[t-1][r]; !ok {
					// No probability from state to observation
					continue
				}
				transitionProbability := V[t-1][r].prob + v.transitionProbabilities[TransitionHash{r, s}]
				if transitionProbability > maxTransitionProbability {
					maxTransitionProbability = transitionProbability
					tmpState = r
				}
			}
			maxProbability := maxTransitionProbability + v.emissionProbabilities[EmissionHash{s, v.observations[t]}]
			V[t][s] = ViterbiVal{prob: maxProbability, prev: tmpState}
		}
	}

	for _, value := range V[len(V)-1] {
		if value.prob > maxPr {
			maxPr = value.prob
		}
	}

	opt := []State{}
	var previous State
	for st, value := range V[len(V)-1] {
		if value.prob == maxPr {
			opt = append(opt, st)
			previous = st
			break
		}
	}
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
