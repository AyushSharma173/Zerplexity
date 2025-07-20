# Deployment Guide for Better-Perplexity App

This guide will help you deploy your Better-Perplexity app to Vercel.

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install with `npm install -g vercel`
3. **Git Repository**: Your code should be in a Git repository

## Project Structure

```
TenexTechnicalTakeHome/
├── backend/                 # FastAPI backend
│   ├── app.py              # Main FastAPI application
│   ├── requirements.txt    # Python dependencies
│   ├── vercel.json        # Vercel configuration
│   └── training_data/     # Training data files
├── frontend/               # React frontend
│   ├── src/               # React source code
│   ├── package.json       # Node.js dependencies
│   ├── vercel.json        # Vercel configuration
│   └── env.example        # Environment variables example
└── deploy.sh              # Deployment script
```

## Step-by-Step Deployment

### Step 1: Prepare Environment Variables

Create a `.env` file in the backend directory with your API keys:

```bash
# Backend Environment Variables
BRAVE_API_KEY=your_brave_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
RERANKER_MODEL=jinaai/jina-reranker-v1-turbo-en
```

### Step 2: Deploy Backend

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Login to Vercel** (if not already logged in):
   ```bash
   vercel login
   ```

3. **Deploy backend**:
   ```bash
   vercel --prod
   ```

4. **Note the backend URL** (you'll need this for the frontend)

### Step 3: Deploy Frontend

1. **Navigate to frontend directory**:
   ```bash
   cd ../frontend
   ```

2. **Create environment file**:
   ```bash
   cp env.example .env
   ```

3. **Update the API URL** in `.env`:
   ```
   VITE_API_URL=https://your-backend-url.vercel.app
   ```

4. **Deploy frontend**:
   ```bash
   vercel --prod
   ```

### Step 4: Configure Environment Variables in Vercel Dashboard

1. Go to your Vercel dashboard
2. Select your backend project
3. Go to Settings → Environment Variables
4. Add the following variables:
   - `BRAVE_API_KEY`
   - `OPENAI_API_KEY`
   - `RERANKER_MODEL`

### Step 5: Test the Deployment

1. Visit your frontend URL
2. Test the chat functionality
3. Verify that API calls work correctly

## Alternative: Use the Deployment Script

You can use the provided deployment script:

```bash
chmod +x deploy.sh
./deploy.sh
```

## Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Update the CORS configuration in `backend/app.py`
   - Add your frontend domain to allowed origins

2. **Environment Variables Not Working**:
   - Check Vercel dashboard for correct variable names
   - Redeploy after adding environment variables

3. **API Calls Failing**:
   - Verify the backend URL is correct in frontend `.env`
   - Check browser console for error messages

4. **Build Failures**:
   - Check `requirements.txt` for all dependencies
   - Ensure Python version compatibility

### Debugging Steps

1. **Check Vercel Logs**:
   ```bash
   vercel logs
   ```

2. **Test Backend Locally**:
   ```bash
   cd backend
   python app.py
   ```

3. **Test Frontend Locally**:
   ```bash
   cd frontend
   npm run dev
   ```

## Environment Variables Reference

### Backend Variables
- `BRAVE_API_KEY`: Your Brave Search API key
- `OPENAI_API_KEY`: Your OpenAI API key
- `RERANKER_MODEL`: Reranker model identifier (default: jinaai/jina-reranker-v1-turbo-en)

### Frontend Variables
- `VITE_API_URL`: URL of your deployed backend

## Post-Deployment

### Monitoring
- Check Vercel dashboard for deployment status
- Monitor API usage and costs
- Set up alerts for errors

### Updates
- Push changes to your Git repository
- Vercel will automatically redeploy
- Or manually trigger deployment with `vercel --prod`

### Scaling
- Vercel automatically scales based on traffic
- Monitor usage in Vercel dashboard
- Consider upgrading plan if needed

## Security Considerations

1. **API Keys**: Never commit API keys to Git
2. **CORS**: Configure CORS properly for production
3. **Rate Limiting**: Consider implementing rate limiting
4. **HTTPS**: Vercel provides HTTPS by default

## Support

- Vercel Documentation: [vercel.com/docs](https://vercel.com/docs)
- FastAPI Documentation: [fastapi.tiangolo.com](https://fastapi.tiangolo.com)
- React Documentation: [reactjs.org](https://reactjs.org) 