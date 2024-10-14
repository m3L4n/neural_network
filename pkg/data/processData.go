package data

import (
	"neural_network/pkg/utils"
	"os"
	"strconv"

	"github.com/go-gota/gota/dataframe"
)

func labelEncoder(label []string) []int {
	var labelBinary = make([]int, len(label))
	for idx, value := range label {
		if value == "M" {
			labelBinary[idx] = 1
		} else {
			labelBinary[idx] = 0
		}
	}
	return labelBinary

}

func normalizedData(data [][]float64) [][]float64 {
	var normalizedData = make([][]float64, len(data))
	var maxValue = utils.MaxMatrix(data)
	var minValue = utils.MinMatrix(data)
	for idx, row := range data {
		normalizedRow := make([]float64, len(row))
		for j, value := range row {

			normalizedRow[j] = ((value - minValue) / (maxValue - minValue))
		}
		normalizedData[idx] = normalizedRow
	}
	return normalizedData

}

// ProcessData take a csv file in parameter , transform it into a df , cut the csv into data and label
//
//	and return a normalized data and the label encoded ( 0 or 1)
func ProcessData(dataset *os.File) ([][]float64, []int, error) {
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
	dataNorm := normalizedData(dataSlice)
	return dataNorm, labelsBinary, nil

}
