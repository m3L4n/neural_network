package data

import (
	"errors"
	"neural_network/pkg/utils"
	"os"
	"strconv"

	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/gonum/mat"
)

func labelEncoder(label []string) mat.Vector {
	var labelVector = mat.NewVecDense(len(label), nil)
	for idx, value := range label {
		if value == "M" {
			labelVector.SetVec(idx, 1.0)
		} else {
			labelVector.SetVec(idx, 0.0)
		}
	}
	return labelVector

}

func normalizedData(data [][]float64) (mat.Matrix, error) {
	var sizeData = len(data)
	if sizeData == 0 {
		return nil, errors.New("Data need to have a positive len")
	}
	var sizeRow = len(data[0])
	if sizeRow == 0 {
		return nil, errors.New("Data need to have a positive len")
	}

	var normalizedData = mat.NewDense(sizeData, sizeRow, nil)
	var maxValue = utils.MaxMatrix(data)
	var minValue = utils.MinMatrix(data)
	for idx, row := range data {
		normalizedRow := make([]float64, len(row))
		for j, value := range row {

			normalizedRow[j] = ((value - minValue) / (maxValue - minValue))
		}
		normalizedData.SetRow(idx, normalizedRow)
	}
	return normalizedData, nil

}

// ProcessData take a csv file in parameter , transform it into a df , cut the csv into data and label
//
//	and return a normalized data and the label encoded ( 0 or 1)
func ProcessData(dataset *os.File) (mat.Matrix, mat.Vector, mat.Matrix, mat.Vector, error) {
	noHeader := dataframe.HasHeader(false)
	var nameColumn = []string{"ID", "Diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension"}
	for len(nameColumn) < 32 {
		nameColumn = append(nameColumn, "f"+strconv.Itoa(32))
	}
	namesColumn := dataframe.Names(nameColumn...)
	var df dataframe.DataFrame = dataframe.ReadCSV(dataset, namesColumn, noHeader)
	dfTrain, dfTest, errTTS := TrainTestSplit(df, 0.2, false)
	if errTTS != nil {
		return nil, nil, nil, nil, errTTS
	}
	var yTrain = dfTrain.Select("Diagnosis")
	var yTest = dfTest.Select("Diagnosis")
	var xTrain = dfTrain.Drop([]string{"ID", "Diagnosis"})
	var xTest = dfTest.Drop([]string{"ID", "Diagnosis"})

	matrixXTrain, err := utils.DfToFloat64Slice(xTrain)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	matrixXTest, err := utils.DfToFloat64Slice(xTest)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	vectorYTrain, errL := utils.DfToStringSlice(yTrain)
	if errL != nil {
		return nil, nil, nil, nil, errL
	}
	vectorYTest, errLT := utils.DfToStringSlice(yTest)
	if errLT != nil {
		return nil, nil, nil, nil, errL
	}
	xTrainNorm, errN := normalizedData(matrixXTrain)
	if errN != nil {
		return nil, nil, nil, nil, errN
	}
	xTestNorm, errN := normalizedData(matrixXTest)
	if errN != nil {
		return nil, nil, nil, nil, errN
	}
	yTrainNorm := labelEncoder(vectorYTrain)
	yTestNorm := labelEncoder(vectorYTest)
	return xTrainNorm, yTrainNorm, xTestNorm, yTestNorm, nil

}
