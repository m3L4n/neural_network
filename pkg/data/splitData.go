package data

import (
	"errors"
	"math/rand"
	"os"

	"github.com/go-gota/gota/dataframe"
)

// TrainTestSplit split a file (csv file) in two part  test and train. it will be cut in two part respecting test size
//
// Usage
// Dataset : *os.File ->  the csv file to split
//
// testSize : float32 between (0.1 and 0.9) ( it will be the representation of test dataset (for example 0.3 => mean that we kept 30% of the len in the test dataset )
//
//	return the two dataset
func TrainTestSplit(dataset *os.File, testSize float32) (dataframe.DataFrame, dataframe.DataFrame, error) {

	if testSize < 0.0 || testSize > 0.9 {
		return dataframe.DataFrame{}, dataframe.DataFrame{}, errors.New("empty name")
	}

	var df dataframe.DataFrame = dataframe.ReadCSV(dataset)
	n := df.Nrow()
	testRows := int(testSize * float32(n))
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
	shuffledTrain := df.Subset(indices[testRows:])
	shuffledTest := df.Subset(indices[:testRows])
	err := SaveData(shuffledTest, "data", "data_train.csv")
	if err != nil {
		return dataframe.DataFrame{}, dataframe.DataFrame{}, err
	}
	errTest := SaveData(shuffledTest, "data", "data_test.csv")
	if errTest != nil {
		return dataframe.DataFrame{}, dataframe.DataFrame{}, errTest
	}
	return shuffledTrain, shuffledTest, nil

}
