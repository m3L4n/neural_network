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

func forward_propagation(X mat.Matrix, weight map[string]mat.Matrix, bias map[string]mat.Vector) map[string]mat.Matrix {
fmt.Print("X SHAPE ",)
	var activations = make(map[string]mat.Matrix, len(weight) + 1)
	activations["A0"] = X
	fmt.Println( X.Dims())
	// sizeWeight := 1
	// for i := range sizeWeight + 1{
		idx := 1 
	// 	if ( idx == sizeWeight + 1){
	// 		break
	// 	}
		// var Z1Tmp mat.Dense
		fmt.Println("idx",idx)
fmt.Print("W SHAPE ",)
		fmt.Println( weight["W" + strconv.Itoa(idx)].Dims())
		fmt.Print("A SHAPE ",)
		fmt.Println(activations["A" + strconv.Itoa(1 - 1)].Dims())
		// Z1Tmp.Mul(activations["A" + strconv.Itoa(idx - 1)], weight["W" + strconv.Itoa(idx)]  )
		// fmt.Print("z SHAPE ",)
		// fmt.Println(Z1Tmp.Dims())     .
		fmt.Print("B SHAPE ",)
		fmt.Println( bias["B" + strconv.Itoa(idx)].Dims(), )

	// }
	// W1 := weight["W1"]
	// b1 := bias["B1"]
	// W2 := weight["W2"]
	// b2 := bias["B2"]
	// var Z1Tmp mat.Dense
	// Z1Tmp.Mul(W1, X)
	// z1, _ := MatrixAddVec(&Z1Tmp, b1)
	// A1 := sigmoid(z1)
	// var Z2Tmp mat.Dense
	// Z2Tmp.Mul(W2, A1)
	// z2, _ := MatrixAddVec(&Z2Tmp, b2)
	// A2 := sigmoid(z2)
	fmt.Println("len of weight", len(weight))
	// activations["A1"] = A1
	// activations["A2"] = A2
	return activations
}

func back_propagation(X, a1, a2 mat.Matrix, weight map[string]mat.Matrix, bias map[string]mat.Vector, y mat.Vector) {
	// m := y.Len()
	// W2 := weight["W2"]

	//   dz2 := a2 - y
	//  dw2 := (1 / m ) * dz2 * a1.T
	//  db2 := (1 / m ) * sum dz2 column  -> need to receive (n,1)
	// dz1 := w2.T * dz2 * a1 * ( 1- a1)
	// dw 1 := (1 / m ) * dz1 * x.T
	//  db1 := (1 / m ) * sum dz1 column  -> need to receive (n,1)
	//  return  dw1, db1, dw2, db2
}

func (n NeuralNetwork) update(dw1, dw2 mat.Matrix, db1, db2 mat.Vector, weight map[string]mat.Matrix, bias map[string]mat.Vector) (map[string]mat.Matrix, map[string]mat.Vector) {
	W1 := weight["W1"]
	W2 := weight["W2"]
	var gradientW1Tmp mat.Dense
	gradientW1Tmp.Scale(n.learningRate, dw1)
	var gradientW1 mat.Dense
	gradientW1.Sub(W1, &gradientW1Tmp)
	var gradientW2Tmp mat.Dense
	gradientW2Tmp.Scale(n.learningRate, dw2)
	var gradientW2 mat.Dense
	gradientW2.Sub(W2, &gradientW2Tmp)

	var newWeight = make(map[string]mat.Matrix, len(weight))
	var newBias = make(map[string]mat.Vector, len(bias))
	newWeight["W1"] = &gradientW1
	newWeight["W2"] = &gradientW2

	b1 := bias["B1"]
	b2 := bias["B2"]

	var gradientB1Tmp mat.VecDense
	gradientB1Tmp.ScaleVec(n.learningRate, db1)
	var gradientB1 mat.VecDense
	gradientB1.SubVec(b1, &gradientB1Tmp)
	var gradientB2Tmp mat.VecDense
	gradientB2Tmp.ScaleVec(n.learningRate, db2)
	var gradientB2 mat.VecDense
	gradientB2.SubVec(b2, &gradientB2Tmp)
	newBias["B1"] = &gradientB1
	newBias["B2"] = &gradientB2
	return newWeight, newBias
}

func predict(X, weight mat.Matrix, bias mat.Vector) {
	model(X, weight, bias)

	return
}

func (n *NeuralNetwork) Fit(XTrain mat.Matrix, YTrain mat.Vector, XTest mat.Matrix, YTest mat.Vector) (*NeuralNetwork, error) {
	red := color.New(color.FgRed, color.Bold).PrintfFunc()
	blue := color.New(color.FgBlue, color.Bold).PrintfFunc()
	XTrain = XTrain.T()
	// YTTrain := YTrain.T()
	XTest = XTest.T()
	// YTTest := YTest.T()
	rowXTrain, columnXtrain := XTrain.Dims()
	rowXTest, columnXtest := XTest.Dims()
	fmt.Printf("x train shape ( %v , %v )\n", rowXTrain, columnXtrain)
	fmt.Printf("x validation  shape ( %v , %v )\n", rowXTest, columnXtest)
	red("Numbers of epoch %v\nLearning rate of %v\nBatch size %v\n", n.epoch, n.learningRate, n.batch)
	for i := range n.epoch {
		blue("Number of epoch %v/%v\n", i, n.epoch)
		forward_propagation(XTrain, n.weight, n.bias)
		// Format(activation["A1"])
		// Format(activation["A2"])

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
