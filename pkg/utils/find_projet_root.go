package utils

import (
	"errors"
	"os"
	"path/filepath"
)

func Find_projet_root() (string, error){

	currentDir, err := os.Getwd()
	if err != nil {
		return "", errors.New("Can't get the current working directory")
	}
		for {
		if _, err := os.Stat(filepath.Join(currentDir, "go.mod")); !os.IsNotExist(err) {
			return currentDir, nil
		}

		parentDir := filepath.Dir(currentDir)
		if parentDir == currentDir {
			return "", errors.New("Could not find project root (missing 'go.mod')")
		}

		currentDir = parentDir
	}
}