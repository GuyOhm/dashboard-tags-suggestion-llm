#!/bin/bash
# Git LFS configuration for the dashboard-tags-suggestion project

echo "🚀 Configuring Git LFS for data files..."

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS is not installed. Installing..."
    
    # Installation based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install git-lfs
        else
            echo "❌ Homebrew required to install Git LFS on macOS"
            echo "📥 Install from: https://git-lfs.github.io/"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get install git-lfs
    else
        echo "❌ OS not automatically supported"
        echo "📥 Install Git LFS from: https://git-lfs.github.io/"
        exit 1
    fi
fi

echo "✅ Git LFS is available"

# Initialize Git LFS in the repository
echo "🔧 Initializing Git LFS..."
git lfs install

# Configure CSV file tracking
echo "📁 Configuring data file tracking..."
git lfs track "data/*.csv"
git lfs track "data/*.json"
git lfs track "data/*.jsonl" 
git lfs track "data/*.parquet"

# Add .gitattributes file
echo "📝 Adding .gitattributes file..."
git add .gitattributes

# Verify tracking
echo "🔍 Verifying LFS tracking:"
git lfs track

echo ""
echo "✅ Git LFS configured successfully!"
echo ""
echo "📋 Next steps:"
echo "   1. git add data/"
echo "   2. git commit -m 'Add datasets with LFS'"
echo "   3. git push"
echo ""
echo "🔍 To verify LFS files:"
echo "   git lfs ls-files"