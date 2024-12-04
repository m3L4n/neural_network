package utils

import (
	"fmt"
	"log"
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


func osToDF(dataset *os.File) dataframe.DataFrame {
	noHeader := dataframe.HasHeader(false)
	var nameColumn = []string{"ID", "Diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension"}
	i := 0
	for len(nameColumn) < 32 {

		nameColumn = append(nameColumn, "f"+strconv.Itoa(i))
		i++
	}
	namesColumn := dataframe.Names(nameColumn...)
	var df dataframe.DataFrame = dataframe.ReadCSV(dataset, namesColumn, noHeader)
	return df
}

func selectFeatured(df dataframe.DataFrame) (t.Tensor, t.Tensor) {

	var y = df.Select("Diagnosis")
	var x = df.Drop([]string{"ID", "Diagnosis", "radius", "perimeter", "area", "f10", "f12", "f13"})
	xTensor, err := DfToTensorFloat64(x)
	handleErrorMsg("Error in transformation of X dataframe to tensor", err)
	yTensor, err := DfToTensorLabel(y)
	handleErrorMsg("Error in transformation of  Y dataframe to tensor", err)
	return xTensor, yTensor
}

func PreprocessData(dataset *os.File, split bool) (t.Tensor, t.Tensor, t.Tensor, t.Tensor) {
	df := osToDF(dataset)
	if split {
		dfTrain, dfTest, errTTS := data.TrainTestSplit(df, 0.2, false)
		handleErrorMsg("error in split dataset", errTTS)
		xTrain, yTrain := selectFeatured(dfTrain)
		xTest, yTest := selectFeatured(dfTest)
		xTrainNorm := ZScoreNormTensor(xTrain)
		xTestNorm := ZScoreNormTensor(xTest)
		return xTrainNorm, yTrain, xTestNorm, yTest
	}
	x, y := selectFeatured(df)
	xNorm := ZScoreNormTensor(x)
	return xNorm, y, nil, nil

}

func handleErrorMsg(msg string, err error) {
	if err != nil {
		log.Fatalf(" error : %v \t%v", msg, err)
		fmt.Println(err)
	}
}
