
GO := go
GOFMT := gofmt
GOLINT := golint

BUILD_DIR := ./bin

SPLIT_BIN := $(BUILD_DIR)/split
TRAIN_BIN := $(BUILD_DIR)/train
validation_BIN := $(BUILD_DIR)/test

GOLINT_PATH := $(shell go env GOPATH)/bin
all: build

build: split train validation


split:
	@echo ">> Building the split application..."
	$(GO) build -o $(SPLIT_BIN) ./cmd/split/main.go

train:
	@echo ">> Building the train application..."
	$(GO) build -o $(TRAIN_BIN) ./cmd/train/main.go

validation:
	@echo ">> Building the est application..."
	$(GO) build -o $(validation_BIN) ./cmd/test/main.go
deps:
	@echo ">> Downloading dependencies..."
	$(GO) mod tidy

test_all:
	@echo ">> Launch test..."
	$(GO) test ./...
lint: install_golint
	@echo ">> Running golint..."
	$(GOLINT) ./cmd/... ./pkg/...
fmt:
	@echo ">> Formatting Go code..."
	$(GOFMT) -w ./cmd/**/*.go ./pkg/**/*.go
clean:
	@echo ">> Cleaning up..."
	rm -rf $(BUILD_DIR)


.PHONY: all build split train validation deps test_all fmt clean lint