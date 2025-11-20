package viterbi

import (
	"errors"
	"fmt"
	"math"
)

// Common errors returned by Viterbi algorithm
var (
	ErrNoObservations    = errors.New("viterbi: no observations provided")
	ErrNoStates          = errors.New("viterbi: no states provided")
	ErrNoStartProbs      = errors.New("viterbi: no start probabilities defined")
	ErrNoValidInitStates = errors.New("viterbi: no valid initial states (check start and emission probabilities)")
	ErrPathBroken        = errors.New("viterbi: path broken - no valid states at observation")
	ErrNoValidPath       = errors.New("viterbi: no valid path found")
	ErrInvalidProbability = errors.New("viterbi: invalid probability value")
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
func (v Viterbi) EvalPath() (ViterbiPath, error) {
	// Check for edge cases
	if len(v.observations) == 0 {
		return ViterbiPath{}, ErrNoObservations
	}
	if len(v.states) == 0 {
		return ViterbiPath{}, ErrNoStates
	}

	var (
		V     = make([]map[State]ViterbiVal, len(v.observations))
		path  map[State][]State
		maxPr = 0.0
	)

	// Initialize first time step
	V[0] = make(map[State]ViterbiVal)
	path = make(map[State][]State)

	// Count valid initial states
	validInitStates := 0
	for s := range v.states {
		st := v.states[s]
		startProb, hasStart := v.startProbabilities[st]
		if !hasStart {
			continue
		}
		// Validate probability value
		if startProb < 0 || startProb > 1 {
			return ViterbiPath{}, fmt.Errorf("%w: start probability %f for state %v", ErrInvalidProbability, startProb, st)
		}
		emissionProb, hasEmission := v.emissionProbabilities[EmissionHash{st, v.observations[0]}]
		if !hasEmission {
			continue
		}
		// Validate emission probability
		if emissionProb < 0 || emissionProb > 1 {
			return ViterbiPath{}, fmt.Errorf("%w: emission probability %f for state %v and observation %v", ErrInvalidProbability, emissionProb, st, v.observations[0])
		}
		V[0][st] = ViterbiVal{prob: startProb * emissionProb}
		path[st] = append(path[st], st)
		validInitStates++
	}

	// Check if we have any valid initial states
	if validInitStates == 0 {
		return ViterbiPath{}, ErrNoValidInitStates
	}

	for t := 1; t < len(v.observations); t++ {
		V[t] = make(map[State]ViterbiVal)
		for s1 := range v.states {
			s := v.states[s1]
			emissionProb, hasEmission := v.emissionProbabilities[EmissionHash{s, v.observations[t]}]
			if !hasEmission {
				// No emission for current state of current observation
				continue
			}
			// Validate emission probability
			if emissionProb < 0 || emissionProb > 1 {
				return ViterbiPath{}, fmt.Errorf("%w: emission probability %f for state %v and observation %v", ErrInvalidProbability, emissionProb, s, v.observations[t])
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
				// Validate transition probability
				if vTransition < 0 || vTransition > 1 {
					return ViterbiPath{}, fmt.Errorf("%w: transition probability %f from state %v to %v", ErrInvalidProbability, vTransition, r, s)
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
			maxProbability := maxTransitionProbability * emissionProb
			V[t][s] = ViterbiVal{prob: maxProbability, prev: tmpState}
		}

		// Check if path breaks at this observation
		if len(V[t]) == 0 {
			return ViterbiPath{}, fmt.Errorf("%w at observation %d", ErrPathBroken, t)
		}
	}

	// Find the maximum probability in the last time step
	if len(V[len(V)-1]) == 0 {
		// No valid path found
		return ViterbiPath{}, ErrNoValidPath
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
		return ViterbiPath{}, ErrNoValidPath
	}

	// Backtrack to find the path (safe - we know all states exist)
	for t := len(V) - 2; t >= 0; t-- {
		if val, ok := V[t+1][previous]; ok {
			opt = append([]State{val.prev}, opt...)
			previous = val.prev
		} else {
			// This should not happen if algorithm is correct, but safety check
			return ViterbiPath{}, fmt.Errorf("viterbi: internal error - backtracking failed at time %d", t+1)
		}
	}

	return ViterbiPath{maxPr, opt}, nil
}

// EvalPathLogProbabilities When every probability is logarithmic
func (v Viterbi) EvalPathLogProbabilities() (ViterbiPath, error) {
	// Check for edge cases
	if len(v.observations) == 0 {
		return ViterbiPath{}, ErrNoObservations
	}
	if len(v.states) == 0 {
		return ViterbiPath{}, ErrNoStates
	}

	var (
		V     = make([]map[State]ViterbiVal, len(v.observations))
		path  map[State][]State
		maxPr = math.Inf(-1)
	)

	// Initialize first time step
	V[0] = make(map[State]ViterbiVal)
	path = make(map[State][]State)

	// Count valid initial states
	validInitStates := 0
	for s := range v.states {
		st := v.states[s]
		startProb, hasStart := v.startProbabilities[st]
		if !hasStart {
			continue
		}
		// For log probabilities, validate they're not positive (log of prob > 1)
		if startProb > 0 {
			return ViterbiPath{}, fmt.Errorf("%w: log start probability %f for state %v should be <= 0", ErrInvalidProbability, startProb, st)
		}
		emissionProb, hasEmission := v.emissionProbabilities[EmissionHash{st, v.observations[0]}]
		if !hasEmission {
			continue
		}
		// Validate log emission probability
		if emissionProb > 0 {
			return ViterbiPath{}, fmt.Errorf("%w: log emission probability %f for state %v and observation %v should be <= 0", ErrInvalidProbability, emissionProb, st, v.observations[0])
		}
		// Check for -Inf values which would break the path
		if math.IsInf(startProb, -1) || math.IsInf(emissionProb, -1) {
			continue // Skip this state but don't error - it's a valid but impossible state
		}
		V[0][st] = ViterbiVal{prob: startProb + emissionProb}
		path[st] = append(path[st], st)
		validInitStates++
	}

	// Check if we have any valid initial states
	if validInitStates == 0 {
		return ViterbiPath{}, ErrNoValidInitStates
	}

	for t := 1; t < len(v.observations); t++ {
		V[t] = make(map[State]ViterbiVal)
		for s1 := range v.states {
			s := v.states[s1]
			emissionProb, hasEmission := v.emissionProbabilities[EmissionHash{s, v.observations[t]}]
			if !hasEmission {
				// No emission for current state of current observation
				continue
			}
			// Validate log emission probability
			if emissionProb > 0 {
				return ViterbiPath{}, fmt.Errorf("%w: log emission probability %f for state %v and observation %v should be <= 0", ErrInvalidProbability, emissionProb, s, v.observations[t])
			}
			// Skip if -Inf (impossible emission)
			if math.IsInf(emissionProb, -1) {
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
				// Validate log transition probability
				if vTransition > 0 {
					return ViterbiPath{}, fmt.Errorf("%w: log transition probability %f from state %v to %v should be <= 0", ErrInvalidProbability, vTransition, r, s)
				}
				stateProb, ok := V[t-1][r]
				if !ok {
					// No probability from state to observation
					continue
				}
				// Skip if either is -Inf
				if math.IsInf(vTransition, -1) || math.IsInf(stateProb.prob, -1) {
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
			maxProbability := maxTransitionProbability + emissionProb
			V[t][s] = ViterbiVal{prob: maxProbability, prev: tmpState}
		}

		// Check if path breaks at this observation
		if len(V[t]) == 0 {
			return ViterbiPath{}, fmt.Errorf("%w at observation %d", ErrPathBroken, t)
		}
	}

	// Find the maximum probability in the last time step
	if len(V[len(V)-1]) == 0 {
		// No valid path found
		return ViterbiPath{}, ErrNoValidPath
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
		return ViterbiPath{}, ErrNoValidPath
	}

	// Backtrack to find the path (safe - we know all states exist)
	for t := len(V) - 2; t >= 0; t-- {
		if val, ok := V[t+1][previous]; ok {
			opt = append([]State{val.prev}, opt...)
			previous = val.prev
		} else {
			// This should not happen if algorithm is correct, but safety check
			return ViterbiPath{}, fmt.Errorf("viterbi: internal error - backtracking failed at time %d", t+1)
		}
	}

	return ViterbiPath{maxPr, opt}, nil
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
