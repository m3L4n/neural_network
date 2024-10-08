
GO := go
GOFMT := gofmt
GOLINT := golint

BUILD_DIR := ./bin

SPLIT_BIN := $(BUILD_DIR)/split
TRAIN_BIN := $(BUILD_DIR)/train
validation_BIN := $(BUILD_DIR)/test

GOLINT_PATH := $(shell go env GOPATH)/bin
all: build

build: split train test


split:
	@echo ">> Building the split application..."
	$(GO) build -o $(SPLIT_BIN) ./cmd/split/main.go

train:
	@echo ">> Building the train application..."
	$(GO) build -o $(TRAIN_BIN) ./cmd/train/main.go

validation:
	@echo ">> Building the est application..."
	$(GO) build -o $(TEST_BIN) ./cmd/test/main.go
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

install_golint:
	@if ! command -v $(GOLINT) &> /dev/null; then \
		echo ">> Installing golint..."; \
		$(GO) install golang.org/x/lint/golint@latest; \
		echo ">> Please add $(GOLINT_PATH) to your PATH."; \
		if [ -f ~/.zshrc ]; then \
			echo 'export PATH=$(PATH):$(GOLINT_PATH)' >> ~/.zshrc; \
		elif [ -f ~/.bashrc ]; then \
			echo 'export PATH=$(PATH):$(GOLINT_PATH)' >> ~/.bashrc; \
		elif [ -f ~/.bash_profile ]; then \
			echo 'export PATH=$(PATH):$(GOLINT_PATH)' >> ~/.bash_profile; \
		fi; \
		echo ">> Added $(GOLINT_PATH) to your shell configuration. Please restart your terminal or run 'source ~/.zshrc'."; \
	fi

.PHONY: all build split train validation deps test_all fmt clean lint