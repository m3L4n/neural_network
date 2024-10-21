package nn

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/fatih/color"
	"gonum.org/v1/gonum/mat"
)

var ErrVectorLengthMismatch = errors.New("matrix rows must match vector length")

// Layer structure represent one layer in the neural network
// it can be a hidden layer or a input layer
type layer struct {
	nNeuron int
	// activationFunction func([]float32) ([]float32, error)
	hiddenLayer bool
	outputLayer bool
	inputLayer  bool
	// weightInitializer  func([]float32) ([]float32, error)
}

// NeuralNetwork structure represent the neural network with his layer,
// and hyperparameter
type NeuralNetwork struct {
	batch        int
	layer        []layer
	learningRate float64
	weight       map[string]mat.Matrix
	bias         map[string]mat.Vector
	data         mat.Matrix
	epoch        int
	// loss         func([]float32, []float32) float32
}

func Format(matrix mat.Matrix) {
	formatted := mat.Formatted(matrix, mat.Prefix(""), mat.Squeeze())
	fmt.Println(formatted)
}
func initWeight(layers []layer) (map[string]mat.Matrix, map[string]mat.Vector, error) {
	nLayers := len(layers)
	if nLayers == 0 {
		return nil, nil, errors.New("Error layer need to have a positive len")
	}
	weight := make(map[string]mat.Matrix, nLayers-1)
	bias := make(map[string]mat.Vector, nLayers-1)
	for idx, value := range layers {
		if idx == 0 {
			continue
		}
		bias["B"+strconv.Itoa(idx)] = mat.NewVecDense(value.nNeuron, nil)
		weight["W"+strconv.Itoa(idx)] = mat.NewDense(value.nNeuron, layers[idx-1].nNeuron, nil)
	}
	return weight, bias, nil
}

// CreateNetwork Permit to create network and initialize the weight for n layer and n neuron in each layer
func (n *NeuralNetwork) CreateNetwork(batch int, layers []int, epoch int, learningRate float64, data mat.Matrix) (*NeuralNetwork, error) {
	layerSize := len(layers)

	layerTmp := make([]layer, layerSize+2)
	rowData, _ := data.Dims()
	layerTmp[0] = layer{
		nNeuron:     rowData,
		hiddenLayer: true,
		outputLayer: false,
		inputLayer:  true,
	}
	for idx, value := range layers {

		layerTmp[idx+1] = layer{
			nNeuron:     value,
			hiddenLayer: true,
			outputLayer: false,
			inputLayer:  false,
		}
	}
	layerTmp[layerSize+1] = layer{
		nNeuron:     2,
		hiddenLayer: false,
		outputLayer: true,
		inputLayer:  false,
	}
	weight, bias, err := initWeight(layerTmp)
	if err != nil {
		return nil, err
	}

	return &NeuralNetwork{
		weight:       weight,
		layer:        layerTmp,
		batch:        batch,
		learningRate: learningRate,
		data:         data,
		bias:         bias,
		epoch:        epoch,
	}, nil

}

func MatrixAddVec(matrix mat.Matrix, vector mat.Vector) (mat.Matrix, error) {
	rowM, columnM := matrix.Dims()
	fmt.Println(rowM, columnM, vector.Len())
	if rowM != vector.Len() {
		fmt.Println(("HELLo"))
		return nil, ErrVectorLengthMismatch
	}
	vRep := mat.NewDense(rowM, columnM, nil)
	var result mat.Dense

	for i := 0; i < vRep.RawMatrix().Cols; i++ {

		vRep.SetCol(i, mat.Col(nil, 0, vector))
	}
	result.Add(matrix, vRep)
	return &result, nil
}

func softmax(Z mat.Vector) mat.Matrix {
	length := Z.Len()
	result := mat.NewVecDense(length, nil)
	prob := mat.NewVecDense(length, nil)
	sum := 0.0
	for i := 0; i < length; i++ {
		tmpExp := math.Exp(Z.AtVec(i))
		sum += tmpExp
		result.SetVec(i, tmpExp)
	}
	for j := 0; j < length; j++ {
		value := result.AtVec(j) / sum
		prob.SetVec(j, value)
	}
	return prob
}
func sigmoid(Z mat.Matrix) mat.Matrix {
	r, c := Z.Dims()
	result := mat.NewDense(r, c, nil)
	result.Apply(func(i, j int, v float64) float64 {
		return 1.0 / (1.0 + math.Exp(-v))
	}, Z)
	return result
}
func model(X mat.Matrix, weight mat.Matrix, bias mat.Vector) (mat.Matrix, error) {
	var C mat.Dense
	C.Mul(weight, X)
	Format(&C)
	Format(bias)
	pred, _ := MatrixAddVec(&C, bias)

	A := sigmoid(pred)
	return A, nil
}

func gradients(a1, a2, X, weight map[string]mat.Matrix, bias map[string]mat.Vector, y mat.Vector) {
	//	sizeLabel := y.Len()
	//
	// dz2 := a2 -y
	// dw2 :=  1/ sizeLabel * (dz2 * a1.T)
	//
	//	db2 := 1/sizeLabel * sum (dz2.col())
	//	dz1 := (weight["W2"].T  * dz2) * a1 * (1 - a1)
	//	dw1 :=  1/ sizeLabel * (dz1 * X.T)
	//
	// db1 := 1/sizeLabel * sum (dz1.col())
	// return dz1, dw1, db1, dz2, dw2, db2
}

func update(dw1, dw2, db1, db2 mat.Matrix, weight map[string]mat.Matrix, bias map[string]mat.Vector) {
	// weight["W1"] =  weight["W1"] - learning_rate * dw1
	// weight["W2"] =  weight["W2"] - learning_rate * dw2
	// bias["B1"] =  bias["B1"] - learning_rate * db1
	// bias["B2"] =  bias["B2"] - learning_rate * db2
	// return weight, bias

}

// func predict(X mat.Matrix,weight, activation mat.Matrix,  bias mat.Vector ){
// 	a, _ := model(X,weight, bias )
// 	return
// }

func (n *NeuralNetwork) Fit(X mat.Matrix, Y mat.Vector) (*NeuralNetwork, error) {
	red := color.New(color.FgRed, color.Bold).PrintfFunc()
	blue := color.New(color.FgBlue, color.Bold).PrintfFunc()
	red("Numbers of epoch %v\nLearning rate of %v\nBatch size %v\n", n.epoch, n.learningRate, n.batch)
	for i := range n.epoch {
		blue("Number of epoch %v/%v\n", i, n.epoch)

	}
	fmt.Println(n.epoch)
	// fmt.Println(n.weight)
	// Format(n.weight["W1"])
	// Format(n.weight["W2"])
	// a1, _ := model(X, n.weight["W1"], n.bias["B1"]) // begin to end
	// Format(a1)
	// a2, _ := model(a1, n.weight["W2"], n.bias["B2"])
	// Format(a2)
	//  back propagation // end to begin
	// update weight and bias
	return n, nil
}
