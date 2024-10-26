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
		fmt.Println("HEELO", idx, value.nNeuron, layers[idx-1].nNeuron)
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
	fmt.Println("ROW data", rowData)
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
	if rowM != vector.Len() {
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
	var activations = make(map[string]mat.Matrix, len(weight)+1)
	activations["A0"] = X
	sizeWeight := len(weight)
	for i := range sizeWeight + 1 {
		idx := i + 1
		if idx == sizeWeight+1 {
			break
		}
		var Z1Tmp mat.Dense
		Z1Tmp.Mul(weight["W"+strconv.Itoa(idx)], activations["A"+strconv.Itoa(idx-1)])
		Z, err := MatrixAddVec(&Z1Tmp, bias["B"+strconv.Itoa(idx)])
		if err != nil {
			fmt.Println("Error on add vector and matrix", err)
		}
		activations["A"+strconv.Itoa(idx)] = sigmoid(Z)
		fmt.Println(activations["A"+strconv.Itoa(idx)].Dims())
	}
	return activations
}

func sumVectorT(vectorT mat.Matrix) (mat.Dense, error) {
	r, _ := vectorT.Dims()
	result := mat.NewDense(r, 1, nil)
	denseMatrix := mat.DenseCopyOf(vectorT)
	for i := range r {
		row := denseMatrix.RowView(i)
		sum := mat.Sum(row)
		result.Set(i, 0, sum)
	}

	return *result, nil
}

func oneHotEncoding(y mat.Matrix) (mat.Matrix, error) {
	_, c := y.Dims()
	newMatrix := mat.NewDense(2, c, nil)
	for i := range c {
		label := y.At(0, i)
		if label == 0.0 {
			newMatrix.Set(0, i, 0.0)
			newMatrix.Set(1, i, 1.0)
		} else if label == 1.0 {
			newMatrix.Set(0, i, 1.0)
			newMatrix.Set(1, i, 0.0)
		} else {
			fmt.Println("Error in the labels, value need to be 1.0 or 0.0")
			return nil, errors.New("Error in the labels, value need to be 1.0 or 0.0")
		}
	}
	return newMatrix, nil
}

func subMatrixScalar(scalar float64, matrix mat.Matrix) mat.Dense {
	r, c := matrix.Dims()
	var sliceScalar = make([]float64, r*c)
	matrixScalar := mat.NewDense(r, c, sliceScalar)
	var result mat.Dense
	result.Sub(matrixScalar, matrix)
	return result
}
func back_propagation(X mat.Matrix, y mat.Matrix, activations map[string]mat.Matrix, weight map[string]mat.Matrix, bias map[string]mat.Vector) (map[string]mat.Matrix, map[string]mat.Matrix) {
	_, c := y.Dims()
	var gradientsWeight = make(map[string]mat.Matrix, len(weight)+1)
	var gradientsBias = make(map[string]mat.Matrix, len(weight)+1)
	sizeWeight := len(weight)
	dzTmp := activations["A"+strconv.Itoa(sizeWeight)]
	var dz mat.Dense
	dz.Sub(dzTmp, y)
	multiplier := 1 / c
	for i := sizeWeight; i > 0; i-- {
		var dw mat.Dense
		dw.Mul(&dz, activations["A"+strconv.Itoa(i-1)].T())
		dw.Scale(float64(multiplier), &dw)
		gradientsWeight["dW"+strconv.Itoa(i)] = &dw
		gradientsBias["dB"+strconv.Itoa(i)] = &dw
		db, errSum := sumVectorT(&dz)
		if errSum != nil {
			fmt.Println("error", nil)
		}
		db.Scale(float64(multiplier), &db)
		if i > 1 {
			var newDzTmp mat.Dense
			newDzTmp.Mul(weight["W"+strconv.Itoa(i)].T(), &dz)
			var newDz mat.Dense
			newDz.Mul(&newDzTmp, activations["A"+strconv.Itoa(i-1)].T())
			subMatrix := subMatrixScalar(1, activations["A"+strconv.Itoa(i-1)])
			var newDz1 mat.Dense
			newDz1.Mul(&newDz, &subMatrix)
			dz = newDz1
		}
	}
	return gradientsWeight, gradientsBias
}

func (n *NeuralNetwork) update(gw, gb map[string]mat.Matrix) {
	c := len(gw)
	for i := range c {
		if ( i == 0){
			continue
		}
		var gd mat.Dense
		gd.Scale(n.learningRate, gw["dW"+strconv.Itoa(i)])
		var newWeight mat.Dense
		newWeight.Sub(n.bias["W"+strconv.Itoa(i)], &gd)
		n.weight["W"+strconv.Itoa(i)] = &newWeight
		var gdB mat.Dense
		gdB.Scale(n.learningRate, gb["dB"+strconv.Itoa(i)])
		var newBias mat.Dense
		newBias.Sub(n.bias["B"+strconv.Itoa(i)], &gdB)
		// n.bias["B" + strconv.Itoa(i)] = &newBias
	}

}

func argMax( a mat.Matrix) mat.Vector{
	r, c := a.Dims()
	var maxArg =   mat.NewVecDense(c, nil)
	for i := range c{
		var max = -1.0
		for row := range r{
			value := a.At(row, i)
			if (max < value){
				max = value
			}
		}
		maxArg.SetVec(i, max)
	}
	return maxArg
}

func predict(X mat.Matrix, weight map[string]mat.Matrix, bias map[string]mat.Vector) mat.Vector {
	  activations := forward_propagation(X, weight, bias)
		finalSize := len(weight)
		aF := activations["A" + strconv.Itoa(finalSize)]
		res := argMax(aF)

	return res
}

func (n *NeuralNetwork) Fit(XTrain mat.Matrix, YTrain mat.Vector, XTest mat.Matrix, YTest mat.Vector) (*NeuralNetwork, error) {
	red := color.New(color.FgRed, color.Bold).PrintfFunc()
	blue := color.New(color.FgBlue, color.Bold).PrintfFunc()
	rowXTrain, columnXtrain := XTrain.Dims()
	rowXTest, columnXtest := XTest.Dims()
	YTrainT := YTrain.T()
	yTrainEncoded, err := oneHotEncoding(YTrainT)
	if err != nil {
		return nil, err
	}
	fmt.Printf("x train shape ( %v , %v )\n", rowXTrain, columnXtrain)
	fmt.Printf("x validation  shape ( %v , %v )\n", rowXTest, columnXtest)
	red("Numbers of epoch %v\nLearning rate of %v\nBatch size %v\n", n.epoch, n.learningRate, n.batch)
	for i := range n.epoch {
		blue("Number of epoch %v/%v\n", i, n.epoch)
		activations := forward_propagation(XTrain, n.weight, n.bias)
		gw, gb := back_propagation(XTrain, yTrainEncoded, activations, n.weight, n.bias)
		n.update(gw, gb)
		

	}
	return n, nil
}
