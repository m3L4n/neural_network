package data

import (
	"fmt"
	"os"
	"testing"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

// Test case when everything works fine

func removeFile(pathFile string) error {
	fmt.Println(pathFile)
	e := os.RemoveAll(pathFile + "/")
	if e == nil {
		fmt.Println("succesfuul delete", pathFile)
	} else {
		fmt.Println("succesfuul delete", e)

	}

	return e

}

func createFolderTest() (string, error) {

	pathRoot, err := FindProjectRoot()
	if err != nil {
		return "", err
	}
	pathFolder := pathRoot + "/test"
	e := os.Mkdir(pathFolder, 0777)

	return pathFolder, e

}
func TestSavedata_Success(t *testing.T) {
	df := dataframe.New(
		series.New([]string{"a", "b", "c", "d", "e"}, series.String, "alphas"),
		series.New([]int{5, 4, 2, 3, 1}, series.Int, "numbers"),
		series.New([]string{"a1", "b2", "c3", "d4", "e5"}, series.String, "alnums"),
		series.New([]bool{true, false, true, true, false}, series.Bool, "state"),
	)

	path, ec := createFolderTest()
	if ec != nil {
		t.Fatalf("Expected no error, but got: %v", ec)
	}
	err := SaveData(df, "test", "test_data.csv")
	if err != nil {
		t.Fatalf("Expected no error, but got: %v", err)
	}
	fmt.Println("HERE THE PATH", path)
	defer removeFile(path)
	// }
	expectedPath := path + "/test_data.csv"
	if _, err := os.Stat(expectedPath); os.IsNotExist(err) {
		t.Fatalf("Expected file to exist at %s but it doesn't", expectedPath)
	}
}

// // Test case when an invalid directory is provided
func TestSavedata_InvalidDirectory(t *testing.T) {
	df := dataframe.New(
		series.New([]string{"a", "b", "c", "d", "e"}, series.String, "alphas"),
		series.New([]int{5, 4, 2, 3, 1}, series.Int, "numbers"),
		series.New([]string{"a1", "b2", "c3", "d4", "e5"}, series.String, "alnums"),
		series.New([]bool{true, false, true, true, false}, series.Bool, "state"),
	)

	err := SaveData(df, "/invalid-directory", "test_data.csv")
	if err == nil {
		t.Fatalf("Expected an error but got nil")
	}

}
