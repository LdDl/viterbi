# viterbi

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Reference](#reference)

## About
Implementation of Viterbi algorithm.
There are two "path" evaluators: [classic](viterbi_test.go#L59) and [logarithmic](viterbi_test.go#L59) (when you don't want underflow).

I prefer to use logarithmic evaluator in applied tasks such as map matching problem (with usage of Hidden Markov Model)

## Installation
```go
go get github.com/LdDl/viterbi
```

## Usage
You can see how to use library in test section: [classic](viterbi_test.go#L26), [logarithmic](viterbi_test.go#L90)


## Reference
https://en.wikipedia.org/wiki/Viterbi_algorithm
