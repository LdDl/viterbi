package viterbi

import "math"

type State interface {
	ID() int
}

type Observation interface {
	ID() int
}

type Viterbi struct {
	states                  []State
	observations            []Observation
	startProbabilities      map[State]float64
	emissionProbabilities   map[State]map[Observation]float64
	transitionProbabilities map[State]map[State]float64
}

func (vi *Viterbi) AddState(state State) {
	vi.states = append(vi.states, state)
}

func (vi *Viterbi) AddObservation(obs Observation) {
	vi.observations = append(vi.observations, obs)
}

func (vi *Viterbi) PutStartProbability(state State, val float64) {
	if vi.startProbabilities == nil {
		vi.startProbabilities = make(map[State]float64)
	}
	vi.startProbabilities[state] = val
}

func (vi *Viterbi) PutEmissionProbability(state State, obs Observation, val float64) {
	if vi.emissionProbabilities == nil {
		vi.emissionProbabilities = make(map[State]map[Observation]float64)
	}
	if _, ok := vi.emissionProbabilities[state]; !ok {
		vi.emissionProbabilities[state] = make(map[Observation]float64)
	}
	vi.emissionProbabilities[state][obs] = val
}

func (vi *Viterbi) PutTransitionProbability(from State, to State, val float64) {
	if vi.transitionProbabilities == nil {
		vi.transitionProbabilities = make(map[State]map[State]float64)
	}
	if _, ok := vi.transitionProbabilities[from]; !ok {
		vi.transitionProbabilities[from] = make(map[State]float64)
	}
	vi.transitionProbabilities[from][to] = val
}

// EvalPath see ref bellow
// https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode
// When every probability is in [0;1]
func (vi *Viterbi) EvalPath() (float64, []State) {
	oLen := len(vi.observations)
	path := make(map[State][]State)
	PbMatrix := make(map[State]map[Observation]float64)
	for _, state := range vi.states {
		PbMatrix[state] = make(map[Observation]float64)
	}
	for _, state := range vi.states {
		if _, ok := vi.startProbabilities[state]; !ok {
			continue
		}
		PbMatrix[state][vi.observations[0]] = vi.startProbabilities[state] * vi.emissionProbabilities[state][vi.observations[0]]
		path[state] = []State{state}
	}
	for i := 1; i < oLen; i++ {
		newPath := make(map[State][]State)
		for _, newState := range vi.states {
			tMaxPb := -math.MaxFloat64
			var tState State
			if _, ok := vi.emissionProbabilities[newState][vi.observations[i]]; !ok {
				// No emission for current state of current observation
				continue
			}
			for _, oldState := range vi.states {
				if _, ok := vi.transitionProbabilities[oldState][newState]; !ok {
					// No transition between states
					continue
				}
				if _, ok := PbMatrix[oldState][vi.observations[i-1]]; !ok {
					// No probability from state to observation
					continue
				}
				tProp := PbMatrix[oldState][vi.observations[i-1]] * vi.transitionProbabilities[oldState][newState] * vi.emissionProbabilities[newState][vi.observations[i]]
				if tProp > tMaxPb {
					tMaxPb = tProp
					tState = oldState
				}
			}
			PbMatrix[newState][vi.observations[i]] = tMaxPb
			newPath[newState] = append(path[tState], newState)
		}
		path = newPath
	}
	tMaxPb := -math.MaxFloat64
	var tState State
	for _, state := range vi.states {
		if _, ok := PbMatrix[state][vi.observations[oLen-1]]; !ok {
			continue
		}
		tProp := PbMatrix[state][vi.observations[oLen-1]]
		if tProp > tMaxPb {
			tMaxPb = tProp
			tState = state
		}
	}
	return tMaxPb, path[tState]
}

// EvalPathLogProbabilities When every probability is logarithmic
func (vi *Viterbi) EvalPathLogProbabilities() (float64, []State) {
	oLen := len(vi.observations)
	path := make(map[State][]State)
	PbMatrix := make(map[State]map[Observation]float64)
	for _, state := range vi.states {
		PbMatrix[state] = make(map[Observation]float64)
	}
	for _, state := range vi.states {
		if _, ok := vi.startProbabilities[state]; !ok {
			continue
		}
		PbMatrix[state][vi.observations[0]] = vi.emissionProbabilities[state][vi.observations[0]]
		path[state] = []State{state}
	}
	for i := 1; i < oLen; i++ {
		newPath := make(map[State][]State)
		for _, newState := range vi.states {
			tMaxPb := -math.MaxFloat64
			var tState State
			if _, ok := vi.emissionProbabilities[newState][vi.observations[i]]; !ok {
				// No emission for current state of current observation
				continue
			}
			for _, oldState := range vi.states {
				if _, ok := vi.transitionProbabilities[oldState][newState]; !ok {
					// No transition between states
					continue
				}
				if _, ok := PbMatrix[oldState][vi.observations[i-1]]; !ok {
					// No probability from state to observation
					continue
				}
				tProp := vi.transitionProbabilities[oldState][newState] + PbMatrix[oldState][vi.observations[i-1]]
				if tProp > tMaxPb {
					tMaxPb = tProp
					tState = oldState
				}
			}
			PbMatrix[newState][vi.observations[i]] = tMaxPb + vi.emissionProbabilities[newState][vi.observations[i]]
			newPath[newState] = append(path[tState], newState)
		}
		path = newPath
	}
	tMaxPb := -math.MaxFloat64
	var tState State
	for _, state := range vi.states {
		if _, ok := PbMatrix[state][vi.observations[oLen-1]]; !ok {
			continue
		}
		tProp := PbMatrix[state][vi.observations[oLen-1]]
		if tProp > tMaxPb {
			tMaxPb = tProp
			tState = state
		}
	}
	return tMaxPb, path[tState]
}
