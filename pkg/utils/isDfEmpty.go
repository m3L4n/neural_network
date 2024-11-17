package utils

import "github.com/go-gota/gota/dataframe"

func isEmpty(df dataframe.DataFrame) bool {
	return df.Nrow() == 0 || df.Ncol() == 0
}
