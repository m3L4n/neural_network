package utils

import (
	"math"
	"neural_network/pkg/data"
	"os"
	"strconv"

	"github.com/go-gota/gota/dataframe"
	t "gorgonia.org/tensor"
)

func MeanTensor(tensor t.Tensor) (float64, error) {
	sum, err := t.Sum(tensor, 0)
	if err != nil {
		return 0, err
	}

	shape := tensor.Shape()
	numElements := shape[0] * shape[1]
	if len(shape) == 1 {
		numElements = shape[0]
	}

	mean := sum.Data().([]float64)[0] / float64(numElements)
	return mean, nil
}

func StdTensor(tensor t.Tensor) (float64, error) {
	mean, err := MeanTensor(tensor)
	if err != nil {
		return 0, err
	}

	squaredDiff := tensor.Clone().(t.Tensor)
	squaredDiff, _ = squaredDiff.Apply(func(x float64) float64 {
		return math.Pow(x-mean, 2)
	})
	if err != nil {
		return 0, err
	}

	variance, err := t.Sum(squaredDiff, 0)
	if err != nil {
		return 0, err
	}

	shape := tensor.Shape()
	numElements := shape[0] * shape[1]
	if len(shape) == 1 {
	}

	varianceValue := variance.Data().([]float64)[0] / float64(numElements)
	stddev := math.Sqrt(varianceValue)

	return stddev, nil
}

func ZScoreNormTensor(tensor t.Tensor) t.Tensor {
	tensorTmp := tensor.Clone().(t.Tensor)
	mean, err := MeanTensor(tensorTmp)
	handleError(err)
	stddev, err := StdTensor(tensorTmp)
	handleError(err)
	normTensor, err := tensorTmp.Apply(func(x float64) float64 {
		return (x - mean) / stddev
	})
	handleError(err)
	return normTensor
}

func OsFileToTensor(dataset *os.File) (t.Tensor, t.Tensor, t.Tensor, t.Tensor) {
	noHeader := dataframe.HasHeader(false)
	var nameColumn = []string{"ID", "Diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension"}
	i := 0
	for len(nameColumn) < 32 {

		nameColumn = append(nameColumn, "f"+strconv.Itoa(i))
		i++
	}
	namesColumn := dataframe.Names(nameColumn...)
	var df dataframe.DataFrame = dataframe.ReadCSV(dataset, namesColumn, noHeader)
	dfTrain, dfTest, errTTS := data.TrainTestSplit(df, 0.15, false)
	handleError(errTTS)

	var yTrain = dfTrain.Select("Diagnosis")
	var yTest = dfTest.Select("Diagnosis")
	var xTrain = dfTrain.Drop([]string{"ID", "Diagnosis", "radius", "perimeter", "area", "f10", "f12", "f13"})
	var xTest = dfTest.Drop([]string{"ID", "Diagnosis", "radius", "perimeter", "area", "f10", "f12", "f13"})
	xTrainTensor, err := DfToTensorFloat64(xTrain)
	handleError(err)
	xTestTenosr, err := DfToTensorFloat64(xTest)
	handleError(err)
	yTrainTensor, err := DfToTensorLabel(yTrain)
	handleError(err)
	yTestTensor, err := DfToTensorLabel(yTest)
	handleError(err)
	normXTrain := ZScoreNormTensor(xTrainTensor)
	// normXTrain := data.NormTensor(xTrainTensor)
	// normXTest := data.NormTensor(xTestTenosr)
	normXTest := ZScoreNormTensor(xTestTenosr)
	return normXTrain, yTrainTensor, normXTest, yTestTensor
}
