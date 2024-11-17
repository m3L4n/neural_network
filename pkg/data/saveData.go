package data

import (
	"os"
	"path/filepath"

	"github.com/go-gota/gota/dataframe"
)

// SaveData save a dataframe into csv.
// Usage
//
// df : Dataframe ->  the dataframe to save in csv
//
// directoryName : string -> the name of the directory at the root of the project
//
// nameFile : string -> the name of the file with its extension (example : x.csv)
func SaveData(df dataframe.DataFrame, directoryName, nameFile string) error {

	path, errPath := FindProjectRoot()
	if errPath != nil {
		return nil
	}
	dataPath := filepath.Join(path, directoryName, nameFile)
	file, err := os.Create(dataPath)
	if err != nil {
		return err
	}
	defer file.Close()
	errCreate := df.WriteCSV(file)
	if errCreate != nil {
		return err
	}

	return nil
}
