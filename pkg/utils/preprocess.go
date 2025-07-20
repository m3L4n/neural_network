package utils

import (
	"fmt"
	"log"
	"neural_network/pkg/data"
	"os"

	"github.com/go-gota/gota/dataframe"
	t "gorgonia.org/tensor"
)



func selectFeature(df dataframe.DataFrame) (t.Tensor, t.Tensor) {

	var y = df.Select("Diagnosis")
	var x = df.Drop([]string{"ID", "Diagnosis"})
	xTensor, err := DfToTensorFloat64(x)
	handleErrorMsg("Error in transformation of X dataframe to tensor", err)
	yTensor, err := DfToTensorLabel(y)
	handleErrorMsg("Error in transformation of  Y dataframe to tensor", err)
	return xTensor, yTensor
}

func PreprocessData(dataset *os.File, split bool) (t.Tensor, t.Tensor, t.Tensor, t.Tensor) {
	df := osToDF(dataset)
	if split {
		dfTrain, dfTest, errTTS := data.TrainTestSplit(df, 0.2, false)
		handleErrorMsg("error in split dataset", errTTS)
		xTrain, yTrain := selectFeature(dfTrain)
		xTest, yTest := selectFeature(dfTest)
		
		xTrainNorm := NormalizeTensorZScore(xTrain)
		xTestNorm := NormalizeTensorZScore(xTest)
		return xTrainNorm, yTrain, xTestNorm, yTest
	}
	x, y := selectFeature(df)
	xNorm := NormalizeTensorZScore(x)
	return xNorm, y, nil, nil

}

func handleErrorMsg(msg string, err error) {
	if err != nil {
		log.Fatalf(" error : %v \t%v", msg, err)
		fmt.Println(err)
	}
}
