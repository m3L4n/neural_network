package utils

import (
	"fmt"
	"testing"

	"github.com/go-gota/gota/dataframe"
	"gonum.org/v1/gonum/mat"
)

func TestDfToStringSlice(t *testing.T) {
	// Test empty DataFrame
		stringData := [][]string{}
	df := dataframe.LoadRecords(stringData)
	_, err := DfToStringSlice(df)
	if err == nil {
		t.Error("Expected an error for empty DataFrame, but got nil")
	}

// 	// Test correct DataFrame
	correctDF := dataframe.LoadRecords([][]string{
		{"Label"},
		{"positive"},
		{"negative"},
		{"positive"},
	})
	result, err := DfToStringSlice(correctDF)
	expected := []string{"Label","positive", "negative", "positive"}
	if err != nil || len(result) != 4 || result[0] != "Label" || result[1] != "positive" || result[2] != "negative" || result[3] != "positive"  {
		t.Errorf("Expected %v, but got %v, error: %v", expected, result, err)
	}
}
func TestDfToFloat64Slice(t *testing.T) {
	// Test empty DataFrame
	emptyDF := dataframe.LoadRecords([][]string{})
	_, err := DfToFloat64Slice(emptyDF)
	if err == nil {
		t.Error("Expected an error for empty DataFrame, but got nil")
	}

	// Test DataFrame with invalid data
	invalidDF := dataframe.LoadRecords([][]string{
		{"1.0", "2.0"},
		{"f32_17", "4.0"},
	})
	_, err = DfToFloat64Slice(invalidDF)
	if err != nil {
		t.Error("Expected no error for DataFrame with invalid data, but got error")
	}

	// Test DataFrame with valid data
	validDF := dataframe.LoadRecords([][]string{
		{"1.0", "2.0"},
		{"3.0", "4.0"},
	})
	result, err := DfToFloat64Slice(validDF)
	fmt.Println(result)
	expected := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}
	if err != nil || len(result) != 2 || result[0][0] != 1.0 || result[1][1] != 4.0 {
		t.Errorf("Expected %v, but got %v, error: %v", expected, result, err)
	}
}

func TestDfToFloat64Matrix(t *testing.T) {
	// Test empty DataFrame
	emptyDF := dataframe.LoadRecords([][]string{})
	_, err := DfToFloat64Matrix(emptyDF)
	if err == nil {
		t.Error("Expected an error for empty DataFrame, but got nil")
	}

	// Test DataFrame with invalid data
	invalidDF := dataframe.LoadRecords([][]string{
		{"1.0", "2.0"},
		{"f32_17", "4.0"},
	})
	_, err = DfToFloat64Matrix(invalidDF)
	if err != nil {
		t.Error("Expected no error for DataFrame with invalid data, but got error")
	}

	// Test DataFrame with valid data
	validDF := dataframe.LoadRecords([][]string{
		{"1.0", "2.0"},
		{"3.0", "4.0"},
	})
	result, err := DfToFloat64Matrix(validDF)
	expected := mat.NewDense(2, 2, []float64{
		1.0, 2.0,
		3.0, 4.0,
	})
	if err != nil || !mat.Equal(result, expected) {
		t.Errorf("Expected %v, but got %v, error: %v", mat.Formatted(expected), mat.Formatted(result), err)
	}
}