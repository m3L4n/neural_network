package utils

import (
	"testing"
)

func TestMinSlicePositive(t *testing.T) {
	slice := []float64{
		43.12, 90.74, 17.58, 37.72, 78.40,
		91.39, 79.30, 23.71, 90.79, 0.34,
		57.35, 16.96, 10.91, 85.74, 39.18,
		2.05, 12.77, 86.48, 34.46, 77.30,
	}
	minValue := 0.34
	myMinValue := MinSlice(slice)
	if minValue != myMinValue {
		t.Fatalf("Error in the min value is %v but MinSlice   says : %v", minValue, myMinValue)
	}
}
func TestMinMatrixPositive(t *testing.T) {
	matrix := [][]float64{
		{49.69, 78.70, 25.29, 37.77, 70.30, 69.59, 12.28, 69.88, 47.02, 59.03},
		{73.45, 32.65, 61.67, 42.89, 15.49, 63.73, 17.37, 98.87, 63.53, 5.91},
		{35.32, 53.96, 16.64, 33.50, 44.50, 30.23, 22.61, 71.71, 25.34, 61.32},
		{58.75, 69.19, 81.44, 41.49, 3.99, 47.61, 22.01, 4.30, 14.52, 1.39},
		{78.95, 43.38, 75.54, 67.33, 22.09, 9.91, 36.99, 98.65, 86.42, 15.41},
		{65.62, 31.01, 70.76, 87.26, 0.21, 33.16, 81.13, 11.07, 64.33, 62.13},
		{78.32, 30.31, 37.39, 3.78, 11.02, 41.22, 34.47, 16.43, 81.54, 68.71},
		{97.75, 25.48, 8.91, 96.47, 71.79, 70.29, 17.43, 54.86, 49.70, 15.48},
		{40.79, 6.17, 2.44, 58.35, 89.18, 36.50, 52.38, 64.16, 40.13, 70.48},
		{11.28, 32.79, 64.16, 29.18, 15.82, 90.09, 0.01, 17.38, 48.20, 78.40},
	}
	minValue := 0.01
	myMinValue := MinMatrix(matrix)
	if minValue != myMinValue {
		t.Fatalf("Error in the min value is %v but MinMatrix   says : %v", minValue, myMinValue)
	}
}
func TestminSliceNegative(t *testing.T) {
	slice := []float64{
		-17.43, -29.84, -49.73, -38.76, -39.17, -51.09, -94.53, -55.42, -80.55, -42.15,
		-63.07, -62.85, -55.69, -72.27, -44.47, -99.05, -30.56, -38.21, -42.47, -59.67,
	}

	minValue := 17.43
	myMinValue := MinSlice(slice)
	if minValue != myMinValue {
		t.Fatalf("Error in the min value is %v but minSlice   says : %v", minValue, myMinValue)
	}
}
func TestMinMatrixNegative(t *testing.T) {
	matrix := [][]float64{
		{-23.51, -36.40, -95.75, -99.14, -88.42, -60.39, -67.39, -16.77, -41.04, -17.56},
		{-71.90, -18.02, -33.67, -16.04, -30.54, -41.60, -77.22, -56.13, -51.65, -37.62},
		{-13.44, -42.28, -40.00, -37.64, -73.64, -31.30, -20.46, -68.19, -34.02, -80.34},
		{-83.88, -34.32, -62.50, -55.54, -55.61, -81.57, -72.47, -62.04, -93.63, -34.61},
		{-93.74, -46.08, -45.94, -22.94, -19.19, -69.42, -59.24, -29.34, -87.33, -12.41},
		{-8.16, -36.60, -68.44, -16.91, -87.76, -64.17, -92.21, -81.08, -11.53, -66.30},
		{-10.19, -15.87, -32.80, -57.05, -25.36, -69.76, -93.83, -20.17, -78.77, -35.97},
		{-55.49, -94.49, -65.46, -83.35, -98.08, -78.62, -48.50, -47.64, -53.27, -81.54},
		{-4.56, -78.71, -43.95, -81.07, -11.54, -16.34, -36.74, -83.64, -56.67, -35.78},
		{-17.38, -86.81, -48.72, -40.04, -53.67, -40.16, -68.11, -77.05, -32.85, -7.22},
	}

	minValue := 4.56
	myMinValue := MinMatrix(matrix)
	if minValue != myMinValue {
		t.Fatalf("Error in the min value is %v but MinMatrix   says : %v", minValue, myMinValue)
	}
}

func TestminSliceEmpty(t *testing.T) {
	slice := []float64{}

	minValue := 0.0
	myMinValue := MinSlice(slice)
	if minValue != myMinValue {
		t.Fatalf("Error in the min value is %v but minSlice   says : %v", minValue, myMinValue)
	}
}
func TestMinMatrixEmpty(t *testing.T) {
	matrix := [][]float64{}

	minValue := 0.0
	myMinValue := MinMatrix(matrix)
	if minValue != myMinValue {
		t.Fatalf("Error in the min value is %v but MinMatrix   says : %v", minValue, myMinValue)
	}
}
func TestminSliceBoth(t *testing.T) {
	slice := []float64{
		35.71, -56.32, 72.44, -92.56, 18.24,
		-33.17, 88.91, -12.77, -95.24, 63.18,
		-49.84, 25.49, -79.53, 31.67, 99.34,
		-45.23, -27.66, 83.71, -91.42, 56.21,
	}

	minValue := 12.77
	myMinValue := MinSlice(slice)
	if minValue != myMinValue {
		t.Fatalf("Error in the min value is %v but minSlice   says : %v", minValue, myMinValue)
	}
}
func TestMinMatrixBoth(t *testing.T) {
	matrix := [][]float64{
		{97.51, -61.42, 23.45, -87.13, 45.32, 63.14, -27.56, 16.98, 34.54, -49.17},
		{-12.23, 54.67, -33.14, 83.77, -72.19, 11.56, 29.18, -88.44, 15.89, 91.27},
		{42.73, -67.43, 89.33, -92.55, 31.89, -57.12, 53.76, -35.64, 10.27, 62.14},
		{-45.39, 92.74, -29.18, 73.45, -12.33, 38.21, -94.88, 61.32, 22.79, -87.44},
		{83.16, -55.87, 96.78, -41.28, 20.45, -70.59, 18.23, 91.05, -17.56, 76.34},
		{-72.14, 55.31, 63.57, -82.21, 25.49, -11.79, 49.53, -97.44, 36.22, -61.32},
		{14.97, -90.16, 83.56, -44.89, 22.71, 89.44, -72.12, 10.67, -67.41, 59.12},
		{-15.82, 70.89, -93.28, 44.73, 81.32, -30.78, 56.74, -89.16, 47.52, 23.68},
		{34.18, -28.75, 49.91, 72.36, -55.72, 11.49, -43.65, 60.23, -94.21, 83.57},
		{17.89, -64.23, 22.34, 66.78, -11.33, 79.34, 37.24, -87.78, 90.32, -22.13},
	}

	minValue := 10.27
	myMinValue := MinMatrix(matrix)
	if minValue != myMinValue {
		t.Fatalf("Error in the min value is %v but MinMatrix   says : %v", minValue, myMinValue)
	}
}
