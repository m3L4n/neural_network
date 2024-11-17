package ml

import (
	"fmt"
	"neural_network/pkg/utils"
	"os"
)

func TrainNN(learningRate float64, dataset *os.File) {
	xTrainTensor, yTraintensor, _, _ := utils.OsFileToTensor(dataset)

	newGd := NewOptimizerGd(learningRate)
	dense1 := NewLayerDense(24, 64)
	activations1 := NewActivation()
	activations2 := NewActivation()
	dense2 := NewLayerDense(64, 24)
	dense3 := NewLayerDense(24, 2)
	lossActivation := NewActivationSoftmax()
	for i := 0; i < 1000; i++ {

		dense1.Foward(xTrainTensor)
		activations1.Forward(dense1.Output)
		dense2.Foward(activations1.Output)
		activations2.Forward(dense2.Output)
		dense3.Foward(activations2.Output)
		lossActivation.Forward(dense3.Output)
		acc := Accuracy(lossActivation.Outpout, yTraintensor)
		loss, _ := BinaryCrossEntropy(lossActivation.Outpout, yTraintensor)
		fmt.Printf(" Epoch %v \t Accuracy : %v \t loss : %v \n", i, acc, loss)
		lossActivation.Backward(lossActivation.Outpout, yTraintensor)
		dense3.Backward(lossActivation.DInput)
		activations2.Backward(dense3.DInput)
		dense2.Backward(dense3.DInput)
		activations1.Backward(dense2.DInput)
		dense1.Backward(activations1.DInput)
		updatedWeight2, updatedBias2 := newGd.UpdateParameter(dense2.Weight, dense2.DWeight, dense2.Bias, dense2.DBias)
		updatedWeight1, updatedBias1 := newGd.UpdateParameter(dense1.Weight, dense1.DWeight, dense1.Bias, dense1.DBias)
		updatedWeight3, updatedBias3 := newGd.UpdateParameter(dense3.Weight, dense3.DWeight, dense3.Bias, dense3.DBias)
		dense2.Weight = updatedWeight2
		dense2.Bias = updatedBias2
		dense1.Weight = updatedWeight1
		dense1.Bias = updatedBias1
		dense3.Weight = updatedWeight3
		dense3.Bias = updatedBias3
	}
}
