package data

import (
	"neural_network/pkg/utils"
	"os"
	"testing"
)

func openDataset() (*os.File, error) {
	path, err := utils.FindProjectRoot()
	if err != nil {
		return &os.File{}, err
	}
	file, err := os.Open(path + "/data/data.csv")

	return file, err
}

func TestSizeOfDF(t *testing.T) {
	file, err := openDataset()
	if err != nil {
		t.Fatalf("Cant open file : %v", err)
	}
	defer file.Close()
	shuffledTrain, shuffledTest, err := TrainTestSplit(file, 0.3)
	if err != nil {
		t.Fatalf("Error in train test split : %v", err)
	}
	totalRows := shuffledTrain.Nrow() + shuffledTest.Nrow()
	if totalRows != 568 {
		t.Fatal("Error the number of line have been altered")
	}

}

func TestSizeTestDataFrame(t *testing.T) {
	file, err := openDataset()
	if err != nil {
		t.Fatalf("Cant open file : %v", err)
	}
	defer file.Close()
	_, shuffledTest, err := TrainTestSplit(file, 0.3)
	if err != nil {
		t.Fatalf("Error in train test split : %v", err)
		totalRows := 568
		expectedTestRows := int(0.3 * float32(totalRows))
		if expectedTestRows != shuffledTest.Nrow() {
			t.Fatal("Error the number of line in the test dataset is not equal to the test size")
		}
	}
}
func TestNoSameElem(t *testing.T) {
	file, err := openDataset()
	if err != nil {
		t.Fatalf("Cant open file : %v", err)
	}
	defer file.Close()
	_, shuffledTest, err := TrainTestSplit(file, 0.3)
	if err != nil {
		t.Fatalf("Error in train test split : %v", err)
		totalRows := 568
		expectedTestRows := int(0.3 * float32(totalRows))
		if expectedTestRows != shuffledTest.Nrow() {
			t.Fatal("Error the number of line in the test dataset is not equal to the test size")
		}
	}
}
