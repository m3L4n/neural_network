package utils

import (
	"errors"
	"log"
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
	fileTrain, err := os.Create("data_train.csv")
	if err != nil {
		log.Fatal(err)
		return dataframe.DataFrame{}, dataframe.DataFrame{}, errors.New("Cant create this file")
	}
	fileTest, err := os.Create("data_test.csv")
	if err != nil {
		log.Fatal(err)
		return dataframe.DataFrame{}, dataframe.DataFrame{}, errors.New("Cant create this file")
	}
	shuffledTrain.WriteCSV(fileTrain)
	shuffledTest.WriteCSV(fileTest)
	return shuffledTrain, shuffledTest, nil

}
