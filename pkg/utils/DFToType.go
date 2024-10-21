package utils

import (
	"errors"
	"log"
	"strconv"

	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/gonum/mat"
)
func isEmpty(df dataframe.DataFrame) bool {
	// Check if the number of rows and columns is 0
	return df.Nrow() == 0 || df.Ncol() == 0
}
// DfToStringSlice take a dataset containing string and return a []string of the dataset
//
//	important because df.Records() return [][]string
func DfToStringSlice(df dataframe.DataFrame) ([]string, error) {
	if isEmpty(df){
		return nil, errors.New("label (diagnosis) of the csv need to be more than 0")
	}
	records := df.Records()
	var dataSlice = make([]string, len(records))
	for idx, row := range records {
		if len(row) == 0 {
			return nil, errors.New("error in the dataset, there is no diagnostic")
		}
		dataSlice[idx] = row[0]

	}
	return dataSlice, nil

}

// DfToFloat64Slice take a dataset containing float64 and return a [][]float64 of the dataset
// important because df.Records return only [][]string
func DfToFloat64Slice(df dataframe.DataFrame) ([][]float64, error) {
		if isEmpty(df){
		return nil, errors.New("features ( different than diagnosis or index ) of the csv need to be more than 0")
	}
	records := df.Records()
	if len(records) == 0 {
		return nil, errors.New("data of the csv need to be more than 0")
	}
	dataSlice := make([][]float64, len(records))

	for idx, row := range records {
		dataRow := make([]float64, len(row))
		for j, value := range row {

			valueFloat, err := strconv.ParseFloat(value, 64)
			if err != nil {
				log.Printf("Error conversion '%s' at the row %d, column %d : %v\n", value, idx, j, err)
				dataRow[j] = 0
				continue
			}
			dataRow[j] = valueFloat
		}

		dataSlice[idx] = dataRow
	}
	return dataSlice, nil
}

// DfToFloat64Matrix take a dataset containing float64 and return a [][]float64 (mat matrix type) of the dataset
// important because df.Records return only [][]string
func DfToFloat64Matrix(df dataframe.DataFrame) (mat.Matrix, error) {
			if isEmpty(df){
		return nil, errors.New("features ( different than diagnosis or index ) of the csv need to be more than 0")
	}
	records := df.Records()
	sizeRecords := len(records)

	if sizeRecords == 0 {
		return nil, errors.New("data of the csv need to be more than 0")
	}
	dataSlice := mat.NewDense(sizeRecords, len(records[0]), nil)

	for idx, row := range records {
		dataRow := make([]float64, len(row))
		for j, value := range row {

			valueFloat, err := strconv.ParseFloat(value, 64)
			if err != nil {
				log.Printf("Error conversion '%s' at the row %d, column %d : %v\n", value, idx, j, err)
				dataRow[j] = 0
				continue
			}
			dataRow[j] = valueFloat
		}
		dataSlice.SetRow(idx, dataRow)
	}
	return dataSlice, nil
}
