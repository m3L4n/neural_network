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
