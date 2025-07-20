#!/bin/bash

echo "🚀 Deploying Better-Perplexity App to Vercel"
echo "=============================================="

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

echo "📁 Current directory: $(pwd)"

# Deploy Backend
echo "🔧 Deploying Backend..."
cd backend
vercel --prod

# Get the backend URL
BACKEND_URL=$(vercel ls | grep -o 'https://[^[:space:]]*' | head -1)
echo "✅ Backend deployed at: $BACKEND_URL"

# Deploy Frontend
echo "🎨 Deploying Frontend..."
cd ../frontend

# Create .env file with backend URL
echo "VITE_API_URL=$BACKEND_URL" > .env

vercel --prod

echo "✅ Deployment complete!"
echo "🌐 Frontend: Check Vercel dashboard for URL"
echo "🔧 Backend: $BACKEND_URL"
echo ""
echo "📝 Next steps:"
echo "1. Set up environment variables in Vercel dashboard"
echo "2. Configure CORS if needed"
echo "3. Test the deployed application" 