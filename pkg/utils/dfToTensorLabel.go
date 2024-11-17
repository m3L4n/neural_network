package utils

import (
	"errors"

	"github.com/go-gota/gota/dataframe"
	t "gorgonia.org/tensor"
)

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
