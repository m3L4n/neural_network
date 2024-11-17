package utils

import (
	"neural_network/pkg/data"
	"os"
	"strconv"

	"github.com/go-gota/gota/dataframe"
	t "gorgonia.org/tensor"
)

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
	dfTrain, dfTest, errTTS := data.TrainTestSplit(df, 0.2, false)
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
	normXTrain := data.NormTensor(xTrainTensor)
	normXTest := data.NormTensor(xTestTenosr)
	return normXTrain, yTrainTensor, normXTest, yTestTensor
}
