#!/bin/bash

# FlyRely API - Local Development Script
# This script sets up and runs the FlyRely API locally on port 8000

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}===============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Main process
print_header "FlyRely API - Local Development Server"

# Step 1: Check and create virtualenv
print_info "Step 1: Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_success "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Step 2: Install requirements
print_info "Step 2: Installing dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
print_success "Dependencies installed"

# Step 3: Copy .env file
print_info "Step 3: Setting up environment variables..."
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
    cp .env.example .env
    print_success ".env file created from .env.example"
elif [ -f ".env" ]; then
    print_success ".env file already exists"
else
    print_info "Creating basic .env file..."
    cat > .env << EOF
WEATHER_API_KEY=Hg4F1R0TRMgVKg3xMorsvg0CzIWXh0ox
WEATHER_API_PROVIDER=tomorrow
EOF
    print_success ".env file created with default values"
fi

# Step 4: Start uvicorn server
print_header "Starting uvicorn server on port 8000..."
print_info "Press Ctrl+C to stop the server"
echo -e "${GREEN}Uvicorn server is starting...${NC}\n"

# Start the server in the background and capture the PID
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
SERVER_PID=$!

# Give the server a moment to start
sleep 2

# Step 5: Open test-page.html in default browser
if [ -f "test-page.html" ]; then
    print_info "Step 5: Opening test page in default browser..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open test-page.html
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        xdg-open test-page.html 2>/dev/null || true
    fi
    print_success "Test page opened in browser"
else
    print_info "Step 5: test-page.html not found, skipping browser open"
fi

print_header "Server is running"
echo -e "${GREEN}FlyRely API is running locally at: http://localhost:8000${NC}"
echo -e "${YELLOW}API documentation available at: http://localhost:8000/docs${NC}\n"

# Wait for the server process
wait $SERVER_PID
