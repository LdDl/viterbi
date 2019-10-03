package viterbi

import (
	"fmt"
	"testing"
)

type CustomState struct {
	Name string
	id   int
}

func (ms CustomState) ID() int {
	return ms.id
}

type CustomObservation struct {
	Name string
	id   int
}

func (mo CustomObservation) ID() int {
	return mo.id
}

func TestViterbiEvalPath(t *testing.T) {
	var (
		incomingStates = []CustomState{
			CustomState{Name: "Healty", id: 1},
			CustomState{Name: "Fever", id: 2},
		}
		incomingObservations = []CustomObservation{
			CustomObservation{Name: "normal", id: 1},
			CustomObservation{Name: "cold", id: 2},
			CustomObservation{Name: "dizzy", id: 3},
		}
	)
	var v Viterbi
	for i := range incomingStates {
		v.AddState(incomingStates[i])
	}
	for i := range incomingObservations {
		v.AddObservation(incomingObservations[i])
	}

	v.PutStartProbability(incomingStates[0], 0.6)
	v.PutStartProbability(incomingStates[1], 0.4)

	v.PutEmissionProbability(incomingStates[0], incomingObservations[0], 0.5)
	v.PutEmissionProbability(incomingStates[0], incomingObservations[1], 0.4)
	v.PutEmissionProbability(incomingStates[0], incomingObservations[2], 0.1)
	v.PutEmissionProbability(incomingStates[1], incomingObservations[0], 0.1)
	v.PutEmissionProbability(incomingStates[1], incomingObservations[1], 0.3)
	v.PutEmissionProbability(incomingStates[1], incomingObservations[2], 0.6)

	v.PutTransitionProbability(incomingStates[0], incomingStates[0], 0.7)
	v.PutTransitionProbability(incomingStates[0], incomingStates[1], 0.3)
	v.PutTransitionProbability(incomingStates[1], incomingStates[0], 0.4)
	v.PutTransitionProbability(incomingStates[1], incomingStates[1], 0.6)

	prob, mostProbablePath := v.EvalPath()

	if len(mostProbablePath) != 3 {
		t.Error(
			"Expected 3 states, but got:", len(mostProbablePath),
		)
	}
	if prob != 0.01512 {
		t.Error(
			"Probability has to be 0.01512, but got", prob,
		)
	}
	if mostProbablePath[0] != incomingStates[0] {
		t.Error(
			"First most probable state has to be 'Healty'm, but got", mostProbablePath[0],
		)
	}
	if mostProbablePath[1] != incomingStates[0] {
		t.Error(
			"Second most probable state has to be 'Healty'm, but got", mostProbablePath[1],
		)
	}
	if mostProbablePath[2] != incomingStates[1] {
		t.Error(
			"Third most probable state has to be 'Fever'm, but got", mostProbablePath[2],
		)
	}
}

func TestViterbiEvalPathLogProbabilities(t *testing.T) {
	var (
		incStates = []CustomState{
			CustomState{Name: "rp11", id: 1}, CustomState{Name: "rp12", id: 2},
			CustomState{Name: "rp21", id: 3}, CustomState{Name: "rp22", id: 4},
			CustomState{Name: "rp31", id: 5}, CustomState{Name: "rp32", id: 6}, CustomState{Name: "rp33", id: 7},
			CustomState{Name: "rp41", id: 8}, CustomState{Name: "rp42", id: 9},
		}
		observations = []CustomObservation{
			CustomObservation{Name: "gps1", id: 1},
			CustomObservation{Name: "gps2", id: 2},
			CustomObservation{Name: "gps3", id: 3},
			CustomObservation{Name: "gps4", id: 4},
		}
	)
	var v Viterbi
	for i := range incStates {
		v.AddState(incStates[i])
	}
	for i := range observations {
		v.AddObservation(observations[i])
	}
	v.PutStartProbability(incStates[0], -5.341012069517231)
	v.PutStartProbability(incStates[1], -77.78334495411055)

	v.PutEmissionProbability(incStates[0], observations[0], -5.341012069517231)
	// fmt.Printf("Emission rp11 (0) for gps1 is %f\n", -5.341012069517231)
	v.PutEmissionProbability(incStates[1], observations[0], -77.78334495411055)
	// fmt.Printf("Emission rp12 (1) for gps1 is %f\n", -77.78334495411055)
	v.PutEmissionProbability(incStates[2], observations[1], -5.341012069517231)
	// fmt.Printf("Emission rp21 (2) for gps2 is %f\n", -5.341012069517231)
	v.PutEmissionProbability(incStates[3], observations[1], -29.488456364381666)
	// fmt.Printf("Emission rp22 (3) for gps2 is %f\n", -29.4884563643816661)
	v.PutEmissionProbability(incStates[4], observations[2], -5.341012069517231)
	// fmt.Printf("Emission rp31 (4) for gps3 is %f\n", -5.341012069517231)
	v.PutEmissionProbability(incStates[5], observations[2], -5.341012069517231)
	// fmt.Printf("Emission rp32 (5) for gps3 is %f\n", -5.341012069517231)
	v.PutEmissionProbability(incStates[6], observations[2], -29.488456364381666)
	// fmt.Printf("Emission rp33 (6) for gps3 is %f\n", -29.488456364381666)
	v.PutEmissionProbability(incStates[7], observations[3], -5.341012069517231)
	// fmt.Printf("Emission rp41 (7) for gps4 is %f\n", -5.341012069517231)
	v.PutEmissionProbability(incStates[8], observations[3], -77.78334495411055)
	// fmt.Printf("Emission rp42 (8) for gps4 is %f\n", -77.78334495411055)

	v.PutTransitionProbability(incStates[0], incStates[2], -1283.6730720901721)
	// fmt.Printf("Transition from rp11 (0)  to rp21 (2) is %f\n", -1283.6730720901721)
	v.PutTransitionProbability(incStates[0], incStates[3], -9129.758656211383)
	// fmt.Printf("Transition from rp12 (1) to rp22 (3) is %f\n", -9129.758656211383)
	v.PutTransitionProbability(incStates[1], incStates[2], -9129.758656211383)
	// fmt.Printf("Transition from rp21 (2) to rp21 (2) is %f\n", -9129.758656211383)
	v.PutTransitionProbability(incStates[1], incStates[3], -1283.6730720901721)
	// fmt.Printf("Transition from rp22 (3) to rp22 (3) is %f\n", -1283.6730720901721)

	v.PutTransitionProbability(incStates[2], incStates[4], 4.646573599499615)
	// fmt.Printf("Transition from rp21 (2) to rp31 (4) is %f\n", 4.646573599499615)
	v.PutTransitionProbability(incStates[2], incStates[5], -2079.898401500611)
	// fmt.Printf("Transition from rp21 (2) to rp32 (5) is %f\n", -2079.898401500611)
	v.PutTransitionProbability(incStates[2], incStates[6], -6248.988351700831)
	// fmt.Printf("Transition from rp21 (2) to rp33 (6) is %f\n", -6248.988351700831)
	v.PutTransitionProbability(incStates[3], incStates[4], -6248.988351700831)
	// fmt.Printf("Transition from rp22 (3) to rp31 (4) is %f\n", -6248.988351700831)
	v.PutTransitionProbability(incStates[3], incStates[5], -4164.443376600721)
	// fmt.Printf("Transition from rp22 (3) to rp32 (5) is %f\n", -4164.443376600721)
	v.PutTransitionProbability(incStates[3], incStates[6], 4.646573599499615)
	// fmt.Printf("Transition from rp22 (3) to rp33 (6) is %f\n", 4.646573599499615)

	v.PutTransitionProbability(incStates[4], incStates[7], -626.5028606174612)
	// fmt.Printf("Transition from rp31 (4) to rp41 (7) is %f\n", -626.5028606174612)
	v.PutTransitionProbability(incStates[4], incStates[8], -3533.29394238376)
	// fmt.Printf("Transition from rp32 (5) to rp42 (8) is %f\n", -3533.29394238376)
	v.PutTransitionProbability(incStates[5], incStates[7], -626.5028606174612)
	// fmt.Printf("Transition from rp33 (6) to rp41 (7) is %f\n", -626.5028606174612)
	v.PutTransitionProbability(incStates[5], incStates[8], -1448.74896728365)
	// fmt.Printf("Transition from rp31 (4) to rp42 (8) is %f\n", -1448.74896728365)
	v.PutTransitionProbability(incStates[6], incStates[7], -3533.29394238376)
	// fmt.Printf("Transition from rp32 (5) to rp41 (7) is %f\n", -3533.29394238376)
	v.PutTransitionProbability(incStates[6], incStates[8], -626.5028606174612)
	// fmt.Printf("Transition from rp33 (6) to rp42 (8) is %f\n", -626.5028606174612)

	prob, path := v.EvalPathLogProbabilities()
	fmt.Println("prob:", prob)
	fmt.Println("path:")
	for i := range path {
		fmt.Println("\t", path[i])
	}

	if prob != -1926.893407386203 {
		t.Error(
			"probability has to be -1926.893407386203, but got", prob,
		)
	}
	if len(path) != 4 {
		t.Error(
			"length of found path has to be 4, but got", len(path),
		)
	}
	if path[0] != incStates[0] {
		t.Error(
			"First state has to be r11, but got", path[0],
		)
	}
	if path[1] != incStates[2] {
		t.Error(
			"Second state has to be r11, but got", path[1],
		)
	}
	if path[2] != incStates[4] {
		t.Error(
			"Third state has to be r31, but got", path[2],
		)
	}
	if path[3] != incStates[7] {
		t.Error(
			"Fourth state has to be r41, but got", path[3],
		)
	}
}
