package nn

import (
	"errors"
	"fmt"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

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
		fmt.Println(idx, value.nNeuron)
		if idx == 0 {
			continue
		}
		bias["B"+strconv.Itoa(idx)] = mat.NewVecDense(value.nNeuron, nil)
		weight["W"+strconv.Itoa(idx)] = mat.NewDense(value.nNeuron, layers[idx-1].nNeuron, nil)
	}
	return weight, bias, nil
}

// CreateNetwork Permit to create network and initialize the weight for n layer and n neuron in each layer
func (n *NeuralNetwork) CreateNetwork(batch int, layers []int, learningRate float64, data mat.Matrix) (*NeuralNetwork, error) {
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
	}, nil

}

func MatrixAddVec(matrix mat.Matrix, vector mat.Vector) mat.Matrix {
	rowM, columnM := matrix.Dims()
	vRep := mat.NewDense(rowM, columnM, nil)
	var result mat.Dense

	for i := 0; i < vRep.RawMatrix().Rows; i++ {
		vRep.SetRow(i, mat.Col(nil, 0, vector))
	}
	result.Add(matrix, vRep)
	return &result
}

func sigmoid(Z mat.Matrix) mat.Matrix {

	var zInv mat.Dense
	zInv.Scale(-1, Z)
	Format(&zInv)
	var matExp mat.Dense
	matExp.Exp(&zInv)
	rowExp, columnExp := matExp.Dims()
	sliceOfOne := make([]float64, rowExp*columnExp)
	for idx := range len(sliceOfOne) {
		sliceOfOne[idx] = 1.
	}
	matOfOne := mat.NewDense(rowExp, columnExp, sliceOfOne)
	var addExp mat.Dense
	addExp.Add(matOfOne, &matExp)
	var sigmoidMatrix mat.Dense
	sigmoidMatrix.DivElem(matOfOne, &addExp)
	Format(&sigmoidMatrix)
	return &sigmoidMatrix
}
func model(X mat.Matrix, weight mat.Matrix, bias mat.Vector) (mat.Matrix, error) {
	var C mat.Dense
	C.Mul(weight, X)
	pred := MatrixAddVec(&C, bias)
	A := sigmoid(pred)
	return A, nil
}

func (n *NeuralNetwork) Fit(X mat.Matrix, Y mat.Vector) (*NeuralNetwork, error) {
	 model(n.weight["W2"], n.weight["W3"], n.bias["B3"]) // begin to end
	//  back propagation // end to begin
	// update weight and bias
	return n, nil
}
