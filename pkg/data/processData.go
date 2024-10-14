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
func ProcessData(dataset *os.File) (mat.Matrix, mat.Vector, error) {
	noHeader := dataframe.HasHeader(false)
	var nameColumn = []string{"ID", "Diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension"}
	for len(nameColumn) < 32 {
		nameColumn = append(nameColumn, "f"+strconv.Itoa(32))
	}
	namesColumn := dataframe.Names(nameColumn...)
	var df dataframe.DataFrame = dataframe.ReadCSV(dataset, namesColumn, noHeader)
	var diagnosis = df.Select("Diagnosis")
	var features = df.Drop([]string{"ID", "Diagnosis"})
	dataSlice, err := utils.DfToFloat64Slice(features)
	if err != nil {
		return nil, nil, err
	}
	labelSlice, errL := utils.DfToStringSlice(diagnosis)
	if errL != nil {
		return nil, nil, errL
	}
	labelsBinary := labelEncoder(labelSlice)
	dataNorm, errN := normalizedData(dataSlice)
	if errN != nil {
		return nil, nil, errN
	}
	return dataNorm, labelsBinary, nil

}
