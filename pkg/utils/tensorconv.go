package utils

import (
	"errors"
	"strconv"

	"github.com/go-gota/gota/dataframe"
	t "gorgonia.org/tensor"
)

func DfToTensorFloat64(df dataframe.DataFrame) (t.Tensor, error) {
	if isEmpty(df) {
		return nil, errors.New("features ( different than diagnosis or index ) of the csv need to be more than 0")
	}
	records := df.Records()
	records = records[1:] // header
	row := len(records)
	column := len(records[0])
	backingTensor := make([]float64, row*column)
	var idxBack = 0
	for _, row := range records {

		for _, value := range row {
			valueFloat, err := strconv.ParseFloat(value, 64)
			handleError(err)
			backingTensor[idxBack] = valueFloat
			idxBack += 1
		}
	}
	dfTensor := t.New(t.WithShape(row, column), t.WithBacking(backingTensor))
	return dfTensor, nil
}

func DfToTensorLabel(df dataframe.DataFrame) (t.Tensor, error) {
	if isEmpty(df) {
		return nil, errors.New("features ( different than diagnosis or index ) of the csv need to be more than 0")
	}
	records := df.Records()
	records = records[1:] //header
	row := len(records)
	backingTensor := make([]float64, row)
	for idx, row := range records {
		for _, value := range row {
			if value == "M" {
				backingTensor[idx] = 1.0
			} else if value == "B" {
				backingTensor[idx] = 0.0
			}
		}
	}
	tensortLabel := t.New(t.WithShape(row, 1), t.WithBacking(backingTensor))
	return tensortLabel, nil
}