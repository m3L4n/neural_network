package data

import (
	"neural_network/pkg/utils"
	"os"
	"path/filepath"

	"github.com/go-gota/gota/dataframe"
)

func Savedata(df dataframe.DataFrame, directory_name, name_file string) (error){

	path , errPath := utils.Find_projet_root()
	if (errPath != nil){
		return nil
	}
	dataPath := filepath.Join(path, directory_name, name_file)
	file, err := os.Create(dataPath)
	if err != nil {
		return err 
	}
	defer file.Close()
	errCreate := df.WriteCSV(file)
		if errCreate != nil {
		return   err 
	}
	
	return nil
}