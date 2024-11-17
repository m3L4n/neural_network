package data

import (
	"errors"
	"os"
	"path/filepath"
)

// FindProjectRoot permit to get the root folder
//
// return the absolute path of the root folder
func FindProjectRoot() (string, error) {

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
