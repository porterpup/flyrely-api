#!/bin/bash

# FlyRely API - Railway Deployment Script
# This script deploys the FlyRely API to Railway with all necessary configurations

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

# Main deployment process
print_header "FlyRely API - Railway Deployment"

# Step 1: Check if railway CLI is installed
print_info "Step 1: Checking Railway CLI installation..."
if ! command -v railway &> /dev/null; then
    print_info "Railway CLI not found. Installing via npm..."
    npm install -g @railway/cli
    print_success "Railway CLI installed successfully"
else
    print_success "Railway CLI is already installed"
fi

# Step 2: Check if logged into Railway
print_info "Step 2: Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    print_info "Not logged into Railway. Opening login..."
    railway login
    print_success "Successfully logged into Railway"
else
    CURRENT_USER=$(railway whoami)
    print_success "Already logged in to Railway as: $CURRENT_USER"
fi

# Step 3: Initialize Railway project
print_info "Step 3: Initializing Railway project..."
if [ ! -f "railway.json" ]; then
    railway init
    print_success "Railway project initialized"
else
    print_info "Railway project already initialized"
fi

# Step 4: Set environment variables
print_info "Step 4: Setting environment variables..."
railway variables set WEATHER_API_KEY=Hg4F1R0TRMgVKg3xMorsvg0CzIWXh0ox
print_success "WEATHER_API_KEY configured"

railway variables set WEATHER_API_PROVIDER=tomorrow
print_success "WEATHER_API_PROVIDER configured"

# Step 5: Deploy to Railway
print_header "Deploying to Railway..."
railway up

# Step 6: Get and display deployment URL
print_header "Deployment Complete"
print_success "Your FlyRely API has been successfully deployed to Railway!"
print_info "Deployment URL information:"
railway status

echo -e "\n${GREEN}The API is now live and ready to use!${NC}\n"
