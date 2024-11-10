package nn

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"neural_network/pkg/utils"
	"strconv"

	"github.com/fatih/color"
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
	epoch        int
	// loss         func([]float32, []float32) float32
}

func Format(matrix mat.Matrix) {
	formatted := mat.Formatted(matrix, mat.Prefix(""), mat.Squeeze())
	fmt.Println(formatted)
}

func initWeight(layers []layer) (map[string]mat.Matrix, map[string]mat.Vector, error) {
	nLayers := len(layers)
	s3 := rand.NewPCG(2, 1)
	r3 := rand.New(s3)

	if nLayers == 0 {
		return nil, nil, errors.New("Error layer need to have a positive len")
	}
	weight := make(map[string]mat.Matrix, nLayers-1)
	bias := make(map[string]mat.Vector, nLayers-1)
	for idx, value := range layers {
		if idx == 0 {
			continue
		}
		variance := 2.0 / float64(layers[idx-1].nNeuron)
		stddev := math.Sqrt(variance)

		var randomValueBias = make([]float64, value.nNeuron)
		for i := 0; i < value.nNeuron; i++ {
			randomValueBias[i] = 0.0
		}
		var randomValueWeight = make([]float64, value.nNeuron*layers[idx-1].nNeuron)
		for i := 0; i < value.nNeuron*layers[idx-1].nNeuron; i++ {
			randomValueWeight[i] = r3.NormFloat64() * stddev
		}
		bias["B"+strconv.Itoa(idx)] = mat.NewVecDense(value.nNeuron, randomValueBias)
		weight["W"+strconv.Itoa(idx)] = mat.NewDense(value.nNeuron, layers[idx-1].nNeuron, randomValueWeight)
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
func ReLU(x mat.Matrix) *mat.Dense {
	r, c := x.Dims()
	result := mat.NewDense(r, c, nil)

	// Appliquer ReLU élément par élément
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			result.Set(i, j, max(0, x.At(i, j)))
		}
	}

	return result
}

// max renvoie le maximum entre deux valeurs.
func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
func forward_propagation(X mat.Matrix, weight map[string]mat.Matrix, bias map[string]mat.Vector) map[string]mat.Matrix {
	var activations = make(map[string]mat.Matrix, len(weight)+1)
	activations["A0"] = X
	sizeWeight := len(weight)
	for idx := 1; idx < sizeWeight+1; idx++ {

		prevActivation := activations["A"+strconv.Itoa(idx-1)]
		W := weight["W"+strconv.Itoa(idx)]
		b := bias["B"+strconv.Itoa(idx)]
		fmt.Println("W")
		Format(W)

		var Z mat.Dense
		Z.Mul(W, prevActivation)
		Z.Apply(func(i, j int, v float64) float64 {
			return v + b.AtVec(i)
		}, &Z)

		fmt.Println("Z")
		Format(&Z)
		if idx == sizeWeight {
			res := utils.Softmax(&Z)
			activations["A"+strconv.Itoa(idx)] = res
		} else {
			activations["A"+strconv.Itoa(idx)] = ReLU(&Z)
		}
	}
	// fmt.Println("after foward", activations)
	return activations
}

func back_propagation(X mat.Matrix, y mat.Matrix, activations map[string]mat.Matrix, weight map[string]mat.Matrix, bias map[string]mat.Vector) (map[string]mat.Matrix, map[string]mat.Matrix) {
	_, c := y.Dims()
	var gradientsWeight = make(map[string]mat.Matrix, len(weight)+1)
	var gradientsBias = make(map[string]mat.Matrix, len(bias)+1)
	sizeWeight := len(weight)
	fmt.Println("cc", sizeWeight, activations)
	var dz mat.Dense
	dz.Sub(activations["A"+strconv.Itoa(sizeWeight)], y)

	for i := sizeWeight; i >= 1; i-- {

		prevActivation := activations["A"+strconv.Itoa(i-1)]

		dW := new(mat.Dense)
		dW.Mul(&dz, prevActivation.T())
		dW.Scale(1/float64(c), dW)
		gradientsWeight["dW"+strconv.Itoa(i)] = dW

		// Calculate gradient bias
		// dbTmp, errSum := utils.SumVectorT(&dz)
		// if errSum != nil {
		// 	fmt.Println("error", nil)
		// }

		var db mat.Dense
		rows, _ := dz.Dims()
		dbData := make([]float64, rows)
		for i := 0; i < rows; i++ {
			dbData[i] = mat.Sum(dz.RowView(i)) / float64(sizeWeight)
		}
		db = *mat.NewDense(rows, 1, dbData)
		gradientsBias["dB"+strconv.Itoa(i)] = &db

		if i > 1 {
			var newDzTmp mat.Dense
			newDzTmp.Mul(weight["W"+strconv.Itoa(i)].T(), &dz)

			var newDz mat.Dense
			newDz.Mul(&newDzTmp, activations["A"+strconv.Itoa(i-1)].T())
			subMatrix := utils.SubMatrixScalar(1, activations["A"+strconv.Itoa(i-1)])
			var newDz1 mat.Dense
			newDz1.Mul(&newDz, &subMatrix)
			dz = newDz1
		}
	}
	// fmt.Println("cc", weight, gradientsWeight, bias, gradientsBias)
	return gradientsWeight, gradientsBias
}

func (n *NeuralNetwork) update(gw, gb map[string]mat.Matrix, weight map[string]mat.Matrix, bias map[string]mat.Vector) (map[string]mat.Matrix, map[string]mat.Vector) {
	c := len(weight)
	newMapW := make(map[string]mat.Matrix, len(weight))
	newMapB := make(map[string]mat.Vector, len(bias))
	for i := 1; i <= c; i++ {

		var gd mat.Dense
		gd.Scale(n.learningRate, gw["dW"+strconv.Itoa(i)])
		var newWeight mat.Dense
		newWeight.Sub(weight["W"+strconv.Itoa(i)], &gd)
		newMapW["W"+strconv.Itoa(i)] = &newWeight

		var gdB mat.Dense
		gdB.Scale(n.learningRate, gb["dB"+strconv.Itoa(i)])
		var newBias mat.Dense
		newBias.Sub(bias["B"+strconv.Itoa(i)], &gdB)
		newMapB["B"+strconv.Itoa(i)] = newBias.ColView(0)
	}
	return newMapW, newMapB
}

func argMax(a mat.Matrix) mat.Vector {
	r, c := a.Dims()
	var maxArg = mat.NewVecDense(c, nil)
	// fmt.Println("Hello in argmax", a)
	// Format(a)
	for i := range c {
		var max = -1.0
		var class = -1.
		for row := range r {

			value := a.At(row, i)
			// fmt.Println("row", row, "value", value)
			if max < value {
				class = float64(row)
				max = value
			}
		}

		maxArg.SetVec(i, class)
	}
	return maxArg
}

func predict(X mat.Matrix, weight map[string]mat.Matrix, bias map[string]mat.Vector) mat.Vector {
	activations := forward_propagation(X, weight, bias)
	finalSize := len(activations) - 1
	aF := activations["A"+strconv.Itoa(finalSize)]
	fmt.Println("AF")
	Format(aF)
	res := argMax(aF)

	return res
}

func (n *NeuralNetwork) Fit(XTrain mat.Matrix, YTrain mat.Vector, XTest mat.Matrix, YTest mat.Vector) (*NeuralNetwork, error) {
	red := color.New(color.FgRed, color.Bold).PrintfFunc()
	blue := color.New(color.FgBlue, color.Bold).PrintfFunc()
	rowXTrain, columnXtrain := XTrain.Dims()
	rowXTest, columnXtest := XTest.Dims()
	YTrainT := YTrain.T()
	yTrainEncoded, err := utils.OneHotEncoding(YTrainT)
	if err != nil {
		return nil, err
	}
	fmt.Printf("x train shape ( %v , %v )\n", rowXTrain, columnXtrain)
	fmt.Printf("x validation  shape ( %v , %v )\n", rowXTest, columnXtest)
	weight := n.weight
	bias := n.bias
	red("Numbers of epoch %v\nLearning rate of %v\nBatch size %v\n", n.epoch, n.learningRate, n.batch)
	for i := range n.epoch {
		blue("Number of epoch %v/%v\n", i, n.epoch)
		activations := forward_propagation(XTrain, weight, bias)
		fmt.Println(activations)
		gw, gb := back_propagation(XTrain, yTrainEncoded, activations, weight, bias)
		fmt.Println("Gradients Weight:", gw)
		fmt.Println("Gradients Bias:", gb)
		weight, bias = n.update(gw, gb, weight, bias)
		// fmt.Println("the weight", weight, bias)
		// // fmt.Println("cc", weight, bias, activations)
		y_pred := predict(XTrain, weight, bias)
		fmt.Println(y_pred)
	}
	return n, nil
}
