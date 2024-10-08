package data

import (
	"errors"
	"math/rand"
	"os"

	"github.com/go-gota/gota/dataframe"
)

func Train_test_split(dataset *os.File, test_size float32) (dataframe.DataFrame, dataframe.DataFrame, error) {

	if test_size < 0.0 || test_size > 0.9 {
		return dataframe.DataFrame{}, dataframe.DataFrame{}, errors.New("empty name")
	}

	var df dataframe.DataFrame = dataframe.ReadCSV(dataset)
	n := df.Nrow()
	testRows := int(test_size * float32(n))
	indices := make([]int, n)
	for i := 0; i < n; i++ {
		indices[i] = i
	}
	rand.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
	shuffledTrain := df.Subset(indices[testRows:])
	shuffledTest := df.Subset(indices[:testRows])
	err := Savedata(shuffledTest, "data", "data_train.csv")
	if (err != nil){
	return dataframe.DataFrame{}, dataframe.DataFrame{}, err
	}
	errTest := Savedata(shuffledTest, "data", "data_test.csv")
	if (errTest != nil){
	return dataframe.DataFrame{}, dataframe.DataFrame{}, errTest
	}
	return shuffledTrain, shuffledTest, nil

}
