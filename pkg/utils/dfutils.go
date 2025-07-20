package utils

import 
(
	"github.com/go-gota/gota/dataframe"
	"os"
	"strconv"
	)

func isEmpty(df dataframe.DataFrame) bool {
	return df.Nrow() == 0 || df.Ncol() == 0
}

func osToDF(dataset *os.File) dataframe.DataFrame {
	noHeader := dataframe.HasHeader(false)
	var nameColumn = []string{"ID", "Diagnosis", "radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension"}
	i := 0
	for len(nameColumn) < 32 {

		nameColumn = append(nameColumn, "f"+strconv.Itoa(i))
		i++
	}
	namesColumn := dataframe.Names(nameColumn...)
	var df dataframe.DataFrame = dataframe.ReadCSV(dataset, namesColumn, noHeader)
	return df
}